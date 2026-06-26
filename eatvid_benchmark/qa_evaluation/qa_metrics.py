"""
问答评估指标模块

提供多种题型的评估指标计算，包括：
- 选择题/判断题准确率
- 填空题准确率（支持容差）
- 简答题BERTScore（预留GPT-4裁判接口）
"""

import logging
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class QAScoreConfig:
    """
    问答评分配置
    
    定义不同难度和类别的权重设置
    """
    # 难度权重（L1基础、L2中等、L3困难）
    difficulty_weights: Dict[str, float] = field(default_factory=lambda: {
        'L1': 1.0,
        'L2': 2.0,
        'L3': 3.0
    })
    
    # 类别权重（behavioral_health更重要）
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        'action_timeline': 1.0,
        'bite': 1.0,
        'facial_expression': 1.0,
        'composition': 1.0,
        'pace': 1.0,
        'metadata': 0.5,
        'behavioral_health': 2.0  # 更高权重
    })
    
    # 题型基础分
    question_type_base_scores: Dict[str, float] = field(default_factory=lambda: {
        'single_choice': 1.0,
        'multi_choice': 2.0,  # 多选题难度更高
        'true_false': 0.5,
        'fill_blank': 1.5,
        'short_answer': 5.0  # 简答题分值最高
    })
    
    # 简答题评估方法
    short_answer_method: str = 'bert_score'  # bert_score, gpt4_judge, key_points
    
    # BERTScore阈值
    bert_score_threshold: float = 0.7
    
    # 是否启用GPT-4裁判（预留接口）
    enable_gpt4_judge: bool = False
    
    def get_question_score(self, difficulty: str, category: str, question_type: str) -> float:
        """计算单个问题的满分"""
        base = self.question_type_base_scores.get(question_type, 1.0)
        diff_weight = self.difficulty_weights.get(difficulty, 1.0)
        cat_weight = self.category_weights.get(category, 1.0)
        return base * diff_weight * cat_weight


class QAMetrics:
    """
    问答评估指标计算器
    
    支持多种题型的评估和综合得分计算。
    """
    
    def __init__(self, config: Optional[QAScoreConfig] = None):
        """
        初始化评估指标计算器
        
        Args:
            config: 评分配置，如果为None则使用默认配置
        """
        self.config = config or QAScoreConfig()
    
    # ==================== 单题评估方法 ====================
    
    def evaluate_single_choice(
        self,
        prediction: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        评估单选题
        
        Args:
            prediction: 模型预测答案（如 "A" 或 "A. xxx"）
            ground_truth: 正确答案（如 "A"）
        
        Returns:
            评估结果字典
        """
        pred_clean = self._extract_option(prediction)
        gt_clean = self._extract_option(ground_truth)
        
        is_correct = pred_clean.upper() == gt_clean.upper()
        
        return {
            'correct': is_correct,
            'score': 1.0 if is_correct else 0.0,
            'prediction_clean': pred_clean,
            'ground_truth_clean': gt_clean
        }
    
    def evaluate_multi_choice(
        self,
        prediction: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        评估多选题
        
        Args:
            prediction: 模型预测答案（如 "A,B,C" 或 "A、B、C"）
            ground_truth: 正确答案（如 "A,B,C"）
        
        Returns:
            评估结果字典
        """
        pred_options = self._extract_multi_options(prediction)
        gt_options = self._extract_multi_options(ground_truth)
        
        # 计算精确匹配
        exact_match = pred_options == gt_options
        
        # 计算部分得分（F1）
        if not pred_options and not gt_options:
            precision = recall = f1 = 1.0
        elif not pred_options or not gt_options:
            precision = recall = f1 = 0.0
        else:
            true_positives = len(pred_options & gt_options)
            precision = true_positives / len(pred_options) if pred_options else 0.0
            recall = true_positives / len(gt_options) if gt_options else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'correct': exact_match,
            'score': 1.0 if exact_match else f1,  # 部分正确给部分分
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'prediction_options': sorted(pred_options),
            'ground_truth_options': sorted(gt_options)
        }
    
    def evaluate_true_false(
        self,
        prediction: str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        评估判断题
        
        Args:
            prediction: 模型预测答案（如 "对" 或 "错"）
            ground_truth: 正确答案（如 "对"）
        
        Returns:
            评估结果字典
        """
        pred_bool = self._parse_boolean(prediction)
        gt_bool = self._parse_boolean(ground_truth)
        
        is_correct = pred_bool == gt_bool
        
        return {
            'correct': is_correct,
            'score': 1.0 if is_correct else 0.0,
            'prediction_bool': pred_bool,
            'ground_truth_bool': gt_bool
        }
    
    def evaluate_fill_blank(
        self,
        prediction: str,
        ground_truth: str,
        tolerance: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        评估填空题
        
        Args:
            prediction: 模型预测答案
            ground_truth: 正确答案
            tolerance: 容差设置，支持：
                - type: "exact" | "range" | "flexible"
                - min, max: 范围容差
                - acceptable_answers: 可接受的答案列表
        
        Returns:
            评估结果字典
        """
        pred_clean = prediction.strip()
        gt_clean = ground_truth.strip()
        
        # 默认精确匹配
        if tolerance is None:
            is_correct = pred_clean == gt_clean
            return {
                'correct': is_correct,
                'score': 1.0 if is_correct else 0.0,
                'prediction_clean': pred_clean,
                'ground_truth_clean': gt_clean
            }
        
        tolerance_type = tolerance.get('type', 'exact')
        
        if tolerance_type == 'exact':
            is_correct = pred_clean == gt_clean
        
        elif tolerance_type == 'range':
            # 数值范围容差
            try:
                pred_num = self._extract_number(pred_clean)
                min_val = tolerance.get('min')
                max_val = tolerance.get('max')
                
                if pred_num is not None and min_val is not None and max_val is not None:
                    is_correct = min_val <= pred_num <= max_val
                else:
                    is_correct = pred_clean == gt_clean
            except (ValueError, TypeError):
                is_correct = False
        
        elif tolerance_type == 'flexible':
            # 灵活匹配
            acceptable = tolerance.get('acceptable_answers', [gt_clean])
            is_correct = pred_clean in acceptable
        
        else:
            is_correct = pred_clean == gt_clean
        
        return {
            'correct': is_correct,
            'score': 1.0 if is_correct else 0.0,
            'prediction_clean': pred_clean,
            'ground_truth_clean': gt_clean,
            'tolerance_applied': tolerance
        }
    
    def evaluate_short_answer(
        self,
        prediction: str,
        reference_answer: str,
        key_points: Optional[List[str]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        评估简答题
        
        Args:
            prediction: 模型预测答案
            reference_answer: 参考答案
            key_points: 关键点列表
            evaluation_config: 评估配置
        
        Returns:
            评估结果字典
        """
        result = {
            'prediction': prediction,
            'reference_answer': reference_answer,
            'bert_score': None,
            'key_point_coverage': None,
            'gpt4_score': None,
            'final_score': 0.0,
            'score': 0.0,      # 与其他题型保持一致的键名
            'correct': False   # 简答题的"正确"标准：得分超过阈值
        }
        
        # 计算BERTScore
        bert_score = self._compute_bert_score(prediction, reference_answer)
        result['bert_score'] = bert_score
        
        # 计算关键点覆盖率
        if key_points:
            coverage = self._compute_key_point_coverage(prediction, key_points)
            result['key_point_coverage'] = coverage
        
        # GPT-4评分（预留接口）
        if self.config.enable_gpt4_judge and evaluation_config and evaluation_config.get('gpt4_judge'):
            gpt4_score = self._compute_gpt4_score(prediction, reference_answer, key_points)
            result['gpt4_score'] = gpt4_score
        
        # 计算最终得分
        # 目前使用BERTScore作为主要指标
        if bert_score is not None:
            threshold = self.config.bert_score_threshold
            if bert_score >= threshold:
                result['final_score'] = 1.0
                result['score'] = 1.0
                result['correct'] = True
            else:
                # 线性缩放
                final = bert_score / threshold
                result['final_score'] = final
                result['score'] = final
                result['correct'] = False
        else:
            # BERTScore 不可用时，使用关键点覆盖率作为得分
            if key_points and result['key_point_coverage'] is not None:
                coverage_score = result['key_point_coverage'].get('coverage', 0.0)
                result['final_score'] = coverage_score
                result['score'] = coverage_score
                result['correct'] = coverage_score >= 0.7  # 70%关键点覆盖视为正确
        
        return result
    
    # ==================== 批量评估方法 ====================
    
    def compute_overall_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        计算总体评估指标
        
        Args:
            results: 所有问题的评估结果列表
        
        Returns:
            总体指标字典
        """
        if not results:
            return {'error': 'No results to compute'}
        
        # 基础统计
        total = len(results)
        correct = sum(1 for r in results if r.get('correct', False))
        
        # 计算加权指标
        # weighted_correct_score: 基于二元 correct（是否答对）× 题目满分，体现"加权准确率"
        # weighted_score: 基于连续 score（答案质量）× 题目满分，体现"加权得分"（包含部分分）
        weighted_correct_score = 0.0  # 基于correct的加权得分
        total_score = 0.0             # 基于score的加权得分（含部分分）
        max_score = 0.0
        
        for r in results:
            q_score = self.config.get_question_score(
                r.get('difficulty', 'L1'),
                r.get('category', 'bite'),
                r.get('question_type', 'single_choice')
            )
            max_score += q_score
            total_score += r.get('score', 0.0) * q_score
            # 加权准确率：只有完全答对才得满分
            if r.get('correct', False):
                weighted_correct_score += q_score
        
        metrics = {
            'total_questions': total,
            'correct_questions': correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'weighted_score': total_score,       # 含部分分的加权总分（如BERTScore）
            'max_score': max_score,
            # 修正：加权准确率基于二元correct（是否答对），而非连续score
            # 这样才能与"准确率"含义一致；对 short_answer 等题目，BERTScore接近1
            # 但correct=False时，加权准确率仍为0，避免混淆
            'weighted_accuracy': weighted_correct_score / max_score if max_score > 0 else 0.0
        }
        
        # 按题型统计
        metrics['by_type'] = self._compute_group_metrics(results, 'question_type')
        
        # 按类别统计
        metrics['by_category'] = self._compute_group_metrics(results, 'category')
        
        # 按难度统计
        metrics['by_difficulty'] = self._compute_group_metrics(results, 'difficulty')
        
        return metrics
    
    def _compute_group_metrics(
        self,
        results: List[Dict[str, Any]],
        group_key: str
    ) -> Dict[str, Dict[str, float]]:
        """按指定键分组计算指标"""
        groups = defaultdict(list)
        for r in results:
            key = r.get(group_key, 'unknown')
            groups[key].append(r)
        
        group_metrics = {}
        for key, group_results in groups.items():
            total = len(group_results)
            correct = sum(1 for r in group_results if r.get('correct', False))
            
            # 计算加权指标
            weighted_correct_score = 0.0  # 基于correct的加权得分
            total_score = 0.0             # 基于score的加权得分（含部分分）
            max_score = 0.0
            for r in group_results:
                q_score = self.config.get_question_score(
                    r.get('difficulty', 'L1'),
                    r.get('category', 'bite'),
                    r.get('question_type', 'single_choice')
                )
                max_score += q_score
                total_score += r.get('score', 0.0) * q_score
                if r.get('correct', False):
                    weighted_correct_score += q_score
            
            group_metrics[key] = {
                'total': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0.0,
                'weighted_score': total_score,
                'max_score': max_score,
                # 修正：加权准确率基于二元correct，与"准确率"含义一致
                'weighted_accuracy': weighted_correct_score / max_score if max_score > 0 else 0.0
            }
        
        return dict(group_metrics)
    
    # ==================== 辅助方法 ====================
    
    def _extract_option(self, answer: str) -> str:
        """提取选项字母（如从 "A. xxx" 提取 "A"）"""
        answer = answer.strip().upper()
        # 匹配单个字母
        match = re.match(r'^([A-F])', answer)
        if match:
            return match.group(1)
        return answer[0] if answer else ''
    
    def _extract_multi_options(self, answer: str) -> set:
        """提取多选题选项"""
        answer = answer.strip().upper()
        # 支持多种分隔符
        options = re.split(r'[,，、\s]+', answer)
        result = set()
        for opt in options:
            match = re.match(r'^([A-F])', opt.strip())
            if match:
                result.add(match.group(1))
        return result
    
    def _parse_boolean(self, answer: str) -> bool:
        """解析布尔答案"""
        answer = answer.strip().lower()
        if answer in ['对', '正确', 'true', 'yes', '是', '√', '✓']:
            return True
        elif answer in ['错', '错误', 'false', 'no', '否', '×', '✗']:
            return False
        return None
    
    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        # 移除百分号等
        text = text.replace('%', '').replace('％', '')
        # 提取数字
        match = re.search(r'[-+]?\d*\.?\d+', text)
        if match:
            return float(match.group())
        return None
    
    def _compute_bert_score(self, prediction: str, reference: str) -> Optional[float]:
        """
        计算BERTScore
        
        注意：需要安装bert-score包
        pip install bert-score
        
        重要：bert_score 默认会从 HuggingFace 下载模型（如 roberta-large），
        在无网络环境下会永久等待导致程序卡死。
        通过设置 TRANSFORMERS_OFFLINE=1 和 HF_DATASETS_OFFLINE=1 强制离线模式，
        若缓存不存在则自动降级到简单相似度计算。
        """
        try:
            from bert_score import score
            import os
            
            # 设置离线模式，防止程序尝试从网络下载模型而卡死
            # 若缓存存在则直接使用，否则抛出异常
            orig_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE')
            orig_hf_offline = os.environ.get('HF_DATASETS_OFFLINE')
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            try:
                P, R, F1 = score([prediction], [reference], lang='zh', verbose=False)
                return float(F1[0])
            finally:
                # 恢复环境变量
                if orig_transformers_offline is None:
                    os.environ.pop('TRANSFORMERS_OFFLINE', None)
                else:
                    os.environ['TRANSFORMERS_OFFLINE'] = orig_transformers_offline
                if orig_hf_offline is None:
                    os.environ.pop('HF_DATASETS_OFFLINE', None)
                else:
                    os.environ['HF_DATASETS_OFFLINE'] = orig_hf_offline
                    
        except ImportError:
            logger.warning("bert-score not installed, using simple similarity")
            return self._simple_similarity(prediction, reference)
        except Exception as e:
            # 包括离线模式下缓存不存在抛出的 OSError/EnvironmentError
            logger.warning(f"BERTScore failed (possibly no cached model): {e}, falling back to simple similarity")
            return self._simple_similarity(prediction, reference)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算（作为BERTScore的后备方案）"""
        # 使用Jaccard相似度
        words1 = set(text1)
        words2 = set(text2)
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def _compute_key_point_coverage(
        self,
        prediction: str,
        key_points: List[str]
    ) -> Dict[str, Any]:
        """计算关键点覆盖率"""
        if not key_points:
            return {'coverage': 1.0, 'matched': [], 'unmatched': []}
        
        matched = []
        unmatched = []
        
        for kp in key_points:
            # 简单的关键词匹配
            if kp.lower() in prediction.lower():
                matched.append(kp)
            else:
                unmatched.append(kp)
        
        coverage = len(matched) / len(key_points) if key_points else 1.0
        
        return {
            'coverage': coverage,
            'matched': matched,
            'unmatched': unmatched,
            'matched_count': len(matched),
            'total_count': len(key_points)
        }
    
    def _compute_gpt4_score(
        self,
        prediction: str,
        reference: str,
        key_points: Optional[List[str]] = None
    ) -> Optional[float]:
        """
        使用GPT-4进行评分（预留接口）
        
        TODO: 实现GPT-4 API调用
        """
        logger.warning("GPT-4 judge not implemented yet")
        return None