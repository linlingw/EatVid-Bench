"""
问答评估任务模块

提供问答式评估的核心逻辑，包括：
- 单问题评估
- 批量评估
- 结果汇总
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .qa_dataset import QAQuestion, QASample
from .qa_metrics import QAMetrics, QAScoreConfig
from .qa_prompts import QAPromptTemplates
from .qa_prompts_advanced import QABatchPromptTemplates, QAEnhancedPromptTemplates
from .qa_prompts_advanced import QABatchPromptTemplates, QAEnhancedPromptTemplates


logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """单个问题的评估结果"""
    question_id: str
    video_id: str
    question_type: str
    difficulty: str
    category: str
    question: str
    prediction: str
    ground_truth: str
    correct: bool
    score: float
    raw_output: str = ""
    evaluation_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class QAEvaluation:
    """问答评估结果汇总"""
    task_name: str = "qa_evaluation"
    num_samples: int = 0
    num_questions: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    per_question_results: List[QAResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_name': self.task_name,
            'num_samples': self.num_samples,
            'num_questions': self.num_questions,
            'metrics': self.metrics,
            'per_question_results': [r.to_dict() for r in self.per_question_results],
            'metadata': self.metadata
        }


class QATask:
    """
    问答评估任务
    
    负责运行问答式评估并计算指标。
    """
    
    def __init__(
        self,
        model,
        config: Optional[Dict[str, Any]] = None,
        score_config: Optional[QAScoreConfig] = None
    ):
        """
        初始化问答任务
        
        Args:
            model: 视觉语言模型实例
            config: 任务配置
            score_config: 评分配置
        """
        self.model = model
        self.config = config or {}
        self.score_config = score_config or QAScoreConfig()
        self.metrics = QAMetrics(self.score_config)
        self.prompt_templates = QAPromptTemplates()
        # 新增：批量模式配置
        self.batch_mode = False
        self.enhanced_mode = False
    
    def build_prompt(self, question: QAQuestion) -> str:
        """为问题构建Prompt"""
        return self.prompt_templates.format_prompt(
            question_type=question.question_type,
            question=question.question,
            options=question.options,
            hints=question.key_points if question.question_type == 'short_answer' else None
        )
    
    def evaluate_question(
        self,
        question: QAQuestion,
        video_path: str,
        video_id: str
    ) -> QAResult:
        """
        评估单个问题
        
        Args:
            question: 问题对象
            video_path: 视频路径
            video_id: 视频ID
        
        Returns:
            QAResult: 评估结果
        """
        # 构建Prompt
        prompt = self.build_prompt(question)
        
        # 准备模型参数
        model_kwargs = {
            'frames': [],  # 空列表，使用video_path
            'prompt': prompt,
            'video_path': video_path
        }

        # 从模型配置中获取参数
        if hasattr(self.model, 'config') and self.model.config:
            model_config = self.model.config
            # 添加fps参数（如果存在）
            if 'fps' in model_config:
                model_kwargs['fps'] = model_config['fps']
            # 添加其他可能的模型特定参数
            if 'video_nframes' in model_config:
                model_kwargs['video_nframes'] = model_config['video_nframes']
            if 'video_sampling_mode' in model_config:
                model_kwargs['video_sampling_mode'] = model_config['video_sampling_mode']

        # 添加问题元数据（用于baseline模型获取问题信息）
        model_kwargs['question_type'] = question.question_type
        model_kwargs['options'] = question.options
        model_kwargs['ground_truth'] = question.answer
        model_kwargs['reference_answer'] = question.reference_answer
        model_kwargs['key_points'] = question.key_points
        model_kwargs['metadata'] = {
            'question_type': question.question_type,
            'options': question.options,
            'ground_truth': question.answer,
            'reference_answer': question.reference_answer,
            'key_points': question.key_points
        }

        # 调用模型
        try:
            model_output = self.model.inference(**model_kwargs)
            raw_output = model_output.raw_text  # 使用正确的属性名 raw_text

            # 移除所有 <think>...</think> 标签及其内容（包括换行）
            cleaned_output = re.sub(r'<think>[\s\S]*?</think>', '', raw_output)
            # 可选：清理多余换行（避免留下空行）
            prediction = re.sub(r'\n{2,}', '\n', cleaned_output).strip()

            # 提取答案部分，去除prompt部分
            # prediction = raw_output.strip()
            # 处理常见的模型输出格式，只保留答案部分
            if "ASSISTANT:" in prediction:
                prediction = prediction.split("ASSISTANT:")[-1].strip()
            elif "ER:" in prediction:
                # 处理LLaVA-NeXT-Video的输出格式
                lines = prediction.split('\n')
                assistant_lines = [line for line in lines if line.strip().startswith("ASSISTANT:")]
                if assistant_lines:
                    prediction = assistant_lines[0].split("ASSISTANT:")[-1].strip()
                else:
                    # 尝试提取最后一行作为答案
                    prediction = lines[-1].strip()
        except Exception as e:
            logger.error(f"Error running model for {question.question_id}: {e}")
            raw_output = f"ERROR: {str(e)}"
            prediction = ""
        
        # 评估答案
        eval_result = self._evaluate_answer(
            question=question,
            prediction=prediction
        )
        
        # 构建结果
        result = QAResult(
            question_id=question.question_id,
            video_id=video_id,
            question_type=question.question_type,
            difficulty=question.difficulty,
            category=question.category,
            question=question.question,
            prediction=prediction,
            ground_truth=question.answer or question.reference_answer or "",
            correct=eval_result.get('correct', False),
            score=eval_result.get('score', 0.0),
            raw_output=raw_output,
            evaluation_details=eval_result,
            metadata={
                'options': question.options,
                'answer_tolerance': question.answer_tolerance,
                'key_points': question.key_points
            }
        )
        
        return result
    
    def _evaluate_answer(
        self,
        question: QAQuestion,
        prediction: str
    ) -> Dict[str, Any]:
        """根据题型评估答案"""
        q_type = question.question_type
        
        if q_type == 'single_choice':
            return self.metrics.evaluate_single_choice(
                prediction=prediction,
                ground_truth=question.answer
            )
        
        elif q_type == 'multi_choice':
            return self.metrics.evaluate_multi_choice(
                prediction=prediction,
                ground_truth=question.answer
            )
        
        elif q_type == 'true_false':
            return self.metrics.evaluate_true_false(
                prediction=prediction,
                ground_truth=question.answer
            )
        
        elif q_type == 'fill_blank':
            return self.metrics.evaluate_fill_blank(
                prediction=prediction,
                ground_truth=question.answer,
                tolerance=question.answer_tolerance
            )
        
        elif q_type == 'short_answer':
            return self.metrics.evaluate_short_answer(
                prediction=prediction,
                reference_answer=question.reference_answer or "",
                key_points=question.key_points,
                evaluation_config=question.evaluation
            )
        
        else:
            logger.warning(f"Unknown question type: {q_type}")
            return {'correct': False, 'score': 0.0}
    
    def _evaluate_sample_single(
        self,
        sample: QASample,
        video_path: str,
        video_id: str
    ) -> List[QAResult]:
        """
        Single question evaluation mode (one call per question)
        
        Args:
            sample: Sample
            video_path: Video path
            video_id: Video ID
        
        Returns:
            List of evaluation results
        """
        results = []
        for question in sample.questions:
            result = self.evaluate_question(
                question=question,
                video_path=video_path,
                video_id=video_id
            )
            results.append(result)
        return results
    
    def run_batch(
        self,
        samples: List[QASample],
        show_progress: bool = True
    ) -> QAEvaluation:
        """
        批量运行评估
        
        Args:
            samples: 样本列表
            show_progress: 是否显示进度
        
        Returns:
            QAEvaluation: 评估结果汇总
        """
        all_results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                sample_iter = tqdm(samples, desc="Evaluating samples")
            except ImportError:
                sample_iter = samples
        else:
            sample_iter = samples
        
        for sample in sample_iter:
            if sample.video_path is None:
                logger.warning(f"Video not found for {sample.video_id}, skipping")
                continue
            
            for question in sample.questions:
                result = self.evaluate_question(
                    question=question,
                    video_path=sample.video_path,
                    video_id=sample.video_id
                )
                all_results.append(result)
        
        # 计算总体指标
        metrics = self.metrics.compute_overall_metrics([
            {
                'correct': r.correct,
                'score': r.score,
                'question_type': r.question_type,
                'category': r.category,
                'difficulty': r.difficulty
            }
            for r in all_results
        ])
        
        evaluation = QAEvaluation(
            task_name="qa_evaluation",
            num_samples=len(samples),
            num_questions=len(all_results),
            metrics=metrics,
            per_question_results=all_results,
            metadata={
                'evaluated_at': datetime.now().isoformat(),
                'score_config': {
                    'difficulty_weights': self.score_config.difficulty_weights,
                    'category_weights': self.score_config.category_weights,
                    'question_type_base_scores': self.score_config.question_type_base_scores
                }
            }
        )
        
        return evaluation
    
    def set_batch_mode(self, enabled: bool = True, batch_by: str = "single"):
        """
        Set batch mode
        
        Args:
            enabled: Whether to enable batch mode
            batch_by: Grouping method, can be "single" (one call per question), "all" (all in one prompt), "difficulty" (by L1/L2/L3), "category" (by task type)
        """
        self.batch_mode = enabled
        self.batch_by = batch_by
    
    def set_enhanced_mode(self, enabled: bool = True, context_level: str = "basic"):
        """
        设置增强输入模式
        
        Args:
            enabled: 是否启用增强模式
            context_level: 上下文详细程度，可选 "basic", "full"
        """
        self.enhanced_mode = enabled
        self.context_level = context_level
    
    def _parse_batch_output(self, output: str, question_ids: List[str]) -> Dict[str, str]:
        """
        解析批量输出的答案
        
        Args:
            output: 模型输出的原始文本
            question_ids: 问题ID列表
        
        Returns:
            问题ID到答案的映射
        """
        result = {}
        
        # 尝试解析JSON
        try:
            # 提取JSON部分
            json_match = re.search(r'\[[\s\S]*\]', output)
            if json_match:
                json_str = json_match.group()
                answers = json.loads(json_str)
                for item in answers:
                    qid = item.get('question_id', '')
                    ans = item.get('answer', '')
                    result[qid] = ans
                return result
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"JSON解析失败，尝试文本匹配: {e}")
        
        # 备选方案：尝试文本匹配
        lines = output.split('\n')
        for line in lines:
            for qid in question_ids:
                if qid in line:
                    # 尝试提取答案
                    parts = line.split(':')
                    if len(parts) > 1:
                        answer = parts[-1].strip().strip('"').strip('\'')
                        result[qid] = answer
        
        return result
    
    def evaluate_sample_batch(
        self,
        sample: QASample,
        video_path: str,
        video_id: str
    ) -> List[QAResult]:
        """
        批量评估单个视频的所有问题（一次模型调用）
        
        Args:
            sample: 样本
            video_path: 视频路径
            video_id: 视频ID
        
        Returns:
            评估结果列表
        """
        # If batch_by is 'single', use the original method
        if getattr(self, 'batch_by', 'single') == 'single':
            return self._evaluate_sample_single(sample, video_path, video_id)
        
        questions = sample.questions
        if not questions:
            return []
        
        # 构建问题列表
        question_data = []
        for q in questions:
            qdata = {
                'id': q.question_id,
                'type': q.question_type,
                'difficulty': q.difficulty,
                'category': q.category,
                'question': q.question,
                'options': q.options
            }
            question_data.append(qdata)
        
        # 构建批量Prompt
        batch_prompt = QABatchPromptTemplates.build_batch_prompt(
            question_data, 
            mode=self.batch_by if hasattr(self, 'batch_by') else 'all'
        )
        
        # 准备模型参数
        model_kwargs = {
            'frames': [],
            'prompt': batch_prompt,
            'video_path': video_path
        }
        
        # 从模型配置中获取参数
        if hasattr(self.model, 'config') and self.model.config:
            model_config = self.model.config
            # 添加fps参数（如果存在）
            if 'fps' in model_config:
                model_kwargs['fps'] = model_config['fps']
            # 添加其他可能的模型特定参数
            if 'video_nframes' in model_config:
                model_kwargs['video_nframes'] = model_config['video_nframes']
            if 'video_sampling_mode' in model_config:
                model_kwargs['video_sampling_mode'] = model_config['video_sampling_mode']
        
        # 调用模型
        try:
            model_output = self.model.inference(**model_kwargs)
            raw_output = model_output.raw_text
        except Exception as e:
            logger.error(f"批量评估失败 for {video_id}: {e}")
            raw_output = ""
            
        # 移除所有 <think>...</think> 标签及其内容（包括换行）
        cleaned_output = re.sub(r'<think>[\s\S]*?</think>', '', raw_output)
        # 可选：清理多余换行（避免留下空行）
        cleaned_output = re.sub(r'\n{2,}', '\n', cleaned_output).strip()
        
        # 解析答案
        question_ids = [q.question_id for q in questions]
        answers = self._parse_batch_output(cleaned_output, question_ids)
        
        # 评估每个问题
        results = []
        for q in questions:
            prediction = answers.get(q.question_id, "")
            
            # 评估答案
            eval_result = self._evaluate_answer(
                question=q,
                prediction=prediction
            )
            
            result = QAResult(
                question_id=q.question_id,
                video_id=video_id,
                question_type=q.question_type,
                difficulty=q.difficulty,
                category=q.category,
                question=q.question,
                prediction=prediction,
                ground_truth=q.answer or q.reference_answer or "",
                correct=eval_result.get('correct', False),
                score=eval_result.get('score', 0.0),
                raw_output=cleaned_output,
                evaluation_details=eval_result
            )
            results.append(result)
        
        return results
    
    def run_batch_mode(
        self,
        samples: List[QASample],
        show_progress: bool = True
    ) -> QAEvaluation:
        """
        批量运行评估（每个视频一次调用）
        
        Args:
            samples: 样本列表
            show_progress: 是否显示进度
        
        Returns:
            评估结果
        """
        # If batch_by is 'single', use the original single question method
        if getattr(self, 'batch_by', 'single') == 'single':
            logger.info("Using single question mode (delegating to run_batch)")
            return self.run_batch(samples, show_progress)
        
        all_results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                sample_iter = tqdm(samples, desc="Evaluating samples (batch)")
            except ImportError:
                sample_iter = samples
        else:
            sample_iter = samples
        
        for sample in sample_iter:
            if sample.video_path is None:
                logger.warning(f"Video not found for {sample.video_id}, skipping")
                continue
            
            # 批量评估
            results = self.evaluate_sample_batch(
                sample=sample,
                video_path=sample.video_path,
                video_id=sample.video_id
            )
            all_results.extend(results)
        
        # 计算总体指标
        metrics = self.metrics.compute_overall_metrics([
            {
                'correct': r.correct,
                'score': r.score,
                'question_type': r.question_type,
                'category': r.category,
                'difficulty': r.difficulty
            }
            for r in all_results
        ])
        
        evaluation = QAEvaluation(
            task_name="qa_evaluation_batch",
            num_samples=len(samples),
            num_questions=len(all_results),
            metrics=metrics,
            per_question_results=all_results,
            metadata={
                'evaluated_at': datetime.now().isoformat(),
                'mode': 'batch',
                'batch_by': self.batch_by if hasattr(self, 'batch_by') else 'all'
            }
        )
        
        return evaluation

    def save_results(
        self,
        evaluation: QAEvaluation,
        output_dir: str,
        chunk_size: Optional[int] = None
    ):
        """
        保存评估结果
        
        Args:
            evaluation: 评估结果
            output_dir: 输出目录
            chunk_size: 如果指定，按此数量的视频分批保存 qa_evaluation_full.json。
                        例如 chunk_size=100 表示每100个视频保存一个文件。
                        qa_metrics.json 和 qa_results.jsonl 总是保存完整版本。
                        设置为 None（默认）时保留原有的单文件存储行为。
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存汇总指标（始终保存为单文件）
        metrics_file = output_path / 'qa_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation.metrics, f, indent=2, ensure_ascii=False)
        
        # 保存详细结果（jsonl格式，始终完整保存，适合流式读取）
        results_file = output_path / 'qa_results.jsonl'
        with open(results_file, 'w', encoding='utf-8') as f:
            for result in evaluation.per_question_results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
        
        if chunk_size is None:
            # 默认：保存完整的 evaluation_full.json（保持向后兼容）
            full_file = output_path / 'qa_evaluation_full.json'
            with open(full_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        else:
            # 分批保存：按 video_id 分组，每 chunk_size 个视频保存一个文件
            # 这样每个 chunk 文件大小可控，适合大量视频
            video_to_results: Dict[str, List] = {}
            for result in evaluation.per_question_results:
                vid = result.video_id
                if vid not in video_to_results:
                    video_to_results[vid] = []
                video_to_results[vid].append(result)
            
            video_ids = list(video_to_results.keys())
            num_chunks = (len(video_ids) + chunk_size - 1) // chunk_size
            
            # 创建 chunks 子目录
            chunks_dir = output_path / 'chunks'
            chunks_dir.mkdir(exist_ok=True)
            
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(video_ids))
                chunk_video_ids = video_ids[start:end]
                
                # 收集该 chunk 的所有结果
                chunk_results = []
                for vid in chunk_video_ids:
                    chunk_results.extend(video_to_results[vid])
                
                # 构建 chunk 的评估字典
                chunk_data = {
                    'chunk_index': chunk_idx,
                    'chunk_range': f'{start+1}-{end}',
                    'video_ids': chunk_video_ids,
                    'num_videos': len(chunk_video_ids),
                    'num_questions': len(chunk_results),
                    'per_question_results': [r.to_dict() for r in chunk_results]
                }
                
                chunk_file = chunks_dir / f'chunk_{chunk_idx+1:04d}_videos_{start+1}-{end}.json'
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path} "
                       f"({num_chunks} chunk files, {chunk_size} videos/chunk)")
            logger.info(f"Chunk files saved to {chunks_dir}")
    
    def generate_report(self, evaluation: QAEvaluation) -> str:
        """
        生成评估报告
        
        Args:
            evaluation: 评估结果
        
        Returns:
            报告文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append("问答式评估报告")
        lines.append("=" * 60)
        lines.append("")
        
        # 总体指标
        metrics = evaluation.metrics
        lines.append("【总体指标】")
        lines.append(f"  总问题数: {metrics.get('total_questions', 0)}")
        lines.append(f"  正确数: {metrics.get('correct_questions', 0)}")
        lines.append(f"  准确率: {metrics.get('accuracy', 0):.2%}")
        lines.append(f"  加权得分: {metrics.get('weighted_score', 0):.2f} / {metrics.get('max_score', 0):.2f}")
        lines.append(f"  加权准确率: {metrics.get('weighted_accuracy', 0):.2%}")
        lines.append("")
        
        # 按题型统计
        if 'by_type' in metrics:
            lines.append("【按题型统计】")
            for q_type, type_metrics in metrics['by_type'].items():
                lines.append(f"  {q_type}:")
                lines.append(f"    准确率: {type_metrics['accuracy']:.2%} ({type_metrics['correct']}/{type_metrics['total']})")
                lines.append(f"    加权准确率: {type_metrics['weighted_accuracy']:.2%}")
            lines.append("")
        
        # 按类别统计
        if 'by_category' in metrics:
            lines.append("【按类别统计】")
            for cat, cat_metrics in metrics['by_category'].items():
                lines.append(f"  {cat}:")
                lines.append(f"    准确率: {cat_metrics['accuracy']:.2%} ({cat_metrics['correct']}/{cat_metrics['total']})")
                lines.append(f"    加权准确率: {cat_metrics['weighted_accuracy']:.2%}")
            lines.append("")
        
        # 按难度统计
        if 'by_difficulty' in metrics:
            lines.append("【按难度统计】")
            for diff, diff_metrics in metrics['by_difficulty'].items():
                lines.append(f"  {diff}:")
                lines.append(f"    准确率: {diff_metrics['accuracy']:.2%} ({diff_metrics['correct']}/{diff_metrics['total']})")
                lines.append(f"    加权准确率: {diff_metrics['weighted_accuracy']:.2%}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)