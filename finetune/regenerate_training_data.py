#!/usr/bin/env python3
"""
训练数据重新生成脚本

功能：
1. 生成 Non-CoT 数据：所有题型使用正确的答案（L3使用reference_answer）
2. 生成 CoT 数据：根据 answer_source 提取相关证据，使用 qwen API 重写成结构化思维链

用法：
python regenerate_training_data.py --mode cot --use_api
python regenerate_training_data.py --mode no_cot
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from openai import OpenAI

# 默认路径配置
DEFAULT_ANNOTATION_ROOT = '/media/nas_data/zxp_data/output'
DEFAULT_COT_INPUT = '/media/nas_data/zxp_data/LlamaFactory/data/train_data.json'
DEFAULT_NOCOT_INPUT = '/media/nas_data/zxp_data/LlamaFactory/data/train_data_no_cot.json'
DEFAULT_COT_OUTPUT = '/media/nas_data/zxp_data/LlamaFactory/data/train_data_cot_regenerated.json'
DEFAULT_NOCOT_OUTPUT = '/media/nas_data/zxp_data/LlamaFactory/data/train_data_no_cot_regenerated.json'


class EvidenceExtractor:
    """从 answer_source 中提取支撑证据"""

    def __init__(self, annotation_root: str):
        self.annotation_root = Path(annotation_root)

    def extract_evidence(self, answer_source: Dict, category: str, video_id: str) -> str:
        """根据 answer_source 提取相关证据"""
        if not answer_source:
            return ""

        file_path = answer_source.get('file', '')

        # 构建完整文件路径
        # file 格式: action_timeline/task15/task15_action_timeline.json
        # 或 behavioral_health_analysis/task15/task15_behavioral_health_analysis.json
        # file_path 已经包含完整的相对路径，直接拼接 annotation_root 即可
        full_path = self.annotation_root / file_path

        # 尝试读取证据文件
        try:
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    evidence_data = json.load(f)
                return self._format_evidence(evidence_data, category, answer_source)
        except Exception as e:
            print(f"Warning: Could not read evidence file {full_path}: {e}")

        return ""

    def _format_evidence(self, evidence_data: Dict, category: str, answer_source: Dict) -> str:
        """格式化证据数据为文本"""
        evidence_parts = []

        if category == 'action_timeline':
            events = evidence_data.get('action_timeline', [])
            time_range = answer_source.get('time_range', [])
            if time_range:
                # 筛选时间范围内的事件
                start_time = self._parse_time(time_range[0])
                end_time = self._parse_time(time_range[1])
                filtered_events = [
                    e for e in events
                    if start_time <= self._parse_time(e.get('start_time', '00:00:00')) <= end_time
                ]
                for event in filtered_events[:5]:
                    st = event.get('start_time', '')
                    et = event.get('end_time', '')
                    evt = event.get('event', '')
                    evidence_parts.append(f"- {st}-{et}: {evt}")
            else:
                # 如果没有时间范围，使用 event 字段
                if 'event' in answer_source:
                    evidence_parts.append(f"- Event: {answer_source['event']}")
                if 'current_action' in answer_source:
                    evidence_parts.append(f"- Current action: {answer_source['current_action']}")
                if 'next_action' in answer_source:
                    evidence_parts.append(f"- Next action: {answer_source['next_action']}")
                if 'evidence' in answer_source:
                    evidence_parts.append(f"- Evidence: {answer_source['evidence']}")
                # 如果都没有，返回前5个事件
                if not evidence_parts:
                    for event in events[:5]:
                        st = event.get('start_time', '')
                        et = event.get('end_time', '')
                        evt = event.get('event', '')
                        evidence_parts.append(f"- {st}-{et}: {evt}")

        elif category == 'chewing':
            if 'total_chews' in evidence_data:
                evidence_parts.append(f"- Total chews: {evidence_data['total_chews']}")
            if 'duration' in evidence_data:
                evidence_parts.append(f"- Duration: {evidence_data['duration']:.1f}s")
            if 'chewing_events' in evidence_data:
                events = evidence_data['chewing_events']
                avg_conf = sum(e.get('confidence', 0) for e in events) / len(events) if events else 0
                evidence_parts.append(f"- Average confidence: {avg_conf:.2f}")

        elif category == 'intake':
            ca = evidence_data.get('comparison_analysis', {})
            if ca:
                foods = ca.get('consumption_by_food', [])
                if foods:
                    for f in foods:
                        ft = f.get('food_type', '?')
                        ratio = f.get('consumption_ratio', 0)
                        evidence_parts.append(f"- {ft}: {ratio*100:.1f}% consumed")

        elif category == 'bite':
            if 'num_bites' in evidence_data:
                evidence_parts.append(f"- Total bites: {evidence_data['num_bites']}")
            if 'bite_events' in evidence_data:
                events = evidence_data['bite_events']
                right = sum(1 for e in events if e.get('hand') == 'Right')
                left = sum(1 for e in events if e.get('hand') == 'Left')
                evidence_parts.append(f"- Hand usage: Right={right}, Left={left}")
            if 'dominant_percentage' in answer_source:
                evidence_parts.append(f"- Dominant hand percentage: {answer_source['dominant_percentage']}%")

        elif category == 'facial_expression':
            if 'facial_expression_timeline' in evidence_data:
                segments = evidence_data['facial_expression_timeline']
                evidence_parts.append(f"- Total segments: {len(segments)}")
                for seg in segments[:3]:
                    st = seg.get('start_time', '')
                    desc = seg.get('description', '')
                    evidence_parts.append(f"- {st}: {desc}")

        elif category == 'composition':
            comp_data = evidence_data.get('composition', {})
            if isinstance(comp_data, dict):
                comp_by_count = comp_data.get('composition_by_count', {})
                if comp_by_count:
                    items = sorted(comp_by_count.items(), key=lambda x: x[1].get('count', 0), reverse=True)
                    dominant = items[0][0] if items else 'unknown'
                    evidence_parts.append(f"- Dominant food type: {dominant}")
                    for food, data in items[:3]:
                        pct = data.get('percentage', 0)
                        evidence_parts.append(f"- {food}: {pct}%")

        elif category == 'pace':
            overall = evidence_data.get('overall_rate', {})
            if overall:
                bpm = overall.get('bites_per_minute', 0)
                avg_iv = overall.get('avg_interval_seconds', 0)
                pace_label = "fast" if bpm > 10 else ("slow" if bpm < 5 else "moderate")
                evidence_parts.append(f"- Overall pace: {bpm:.1f} bites/min ({pace_label})")
                evidence_parts.append(f"- Average interval: {avg_iv:.1f}s")

        elif category == 'behavioral_health':
            bha = evidence_data.get('behavioral_health_analysis', {})
            if bha:
                if 'eating_speed' in bha:
                    speed = bha['eating_speed']
                    evidence_parts.append(f"- Eating speed: {speed.get('level', 'unknown')} (confidence: {speed.get('confidence', 'unknown')})")
                if 'food_choice_quality' in bha:
                    food = bha['food_choice_quality']
                    evidence_parts.append(f"- Food quality: {food.get('level', 'unknown')}")
                if 'emotional_eating_intensity' in bha:
                    emotion = bha['emotional_eating_intensity']
                    evidence_parts.append(f"- Emotional eating: {emotion.get('level', 'unknown')}")
                if 'psychological_health_indicators' in bha:
                    psych = bha['psychological_health_indicators']
                    evidence_parts.append(f"- Stress level: {psych.get('stress_level', 'unknown')}")
                    evidence_parts.append(f"- Self-control: {psych.get('self_control', 'unknown')}")

        elif category == 'metadata':
            # 提取元数据信息
            if 'duration_seconds' in answer_source:
                dur = answer_source['duration_seconds']
                minutes = int(dur // 60)
                seconds = int(dur % 60)
                evidence_parts.append(f"- Video duration: {minutes} minutes {seconds} seconds ({dur:.2f}s total)")
            if 'video_id' in evidence_data:
                evidence_parts.append(f"- Video ID: {evidence_data['video_id']}")
            if 'fps' in evidence_data:
                evidence_parts.append(f"- FPS: {evidence_data['fps']}")

        return "\n".join(evidence_parts) if evidence_parts else ""

    def _parse_time(self, time_str: str) -> float:
        """解析时间字符串为秒数"""
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except:
            pass
        return 0.0


class CoTReWriter:
    """使用 qwen API 重写思维链"""

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen3.6-flash", enable_thinking: bool = False):
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        self.model = model
        self.enable_thinking = enable_thinking

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def rewrite_reasoning(
        self,
        question: str,
        evidence: str,
        category: str,
        difficulty: str,
        reference_answer: str = ""
    ) -> str:
        """重写思维链"""
        messages = [
            {
                "role": "system",
                "content": "You are a professional video behavior analysis assistant. You need to use a structured chain of thought to answer questions based on observed evidence."
            },
            {
                "role": "user",
                "content": self._build_prompt(question, evidence, category, difficulty, reference_answer)
            }
        ]

        try:
            # 构建 API 调用参数
            call_params = {
                "model": self.model,
                "messages": messages,
            }

            # 如果启用深度思考
            if self.enable_thinking:
                call_params["extra_body"] = {"enable_thinking": True}

            response = self.client.chat.completions.create(**call_params)

            message = response.choices[0].message

            # 如果启用了深度思考且有 reasoning_content，使用 reasoning_content
            if self.enable_thinking and hasattr(message, "reasoning_content") and message.reasoning_content:
                content = message.reasoning_content
            else:
                content = message.content

            # 移除可能的 thinking 标签
            if content.startswith('<think>'):
                content = content.split('</think>')[-1] if '</think>' in content else content
            return content.strip()
        except Exception as e:
            print(f"API调用异常: {e}")
            return self._fallback_cot(question, evidence, category, reference_answer)

    def _build_prompt(
        self,
        question: str,
        evidence: str,
        category: str,
        difficulty: str,
        reference_answer: str
    ) -> str:
        """构建提示词"""
        prompt = f"""You are a video understanding task assistant. Based on the provided evidence, generate a structured chain of thought to answer the question.

Task category: {category}
Difficulty level: {difficulty}

Question:
{question}

Relevant evidence from video analysis:
{evidence}
"""
        if reference_answer:
            prompt += f"\nReference answer (for your understanding, do NOT copy directly):\n{reference_answer}\n"

        prompt += """
Please organize your chain of thought as follows:

1. **Task Analysis**: Clarify what this question needs to focus on for this specific category
2. **Evidence Observation**: Extract key observations from the evidence above
3. **Conclusion**: Derive the answer based on observations

Requirements:
- Only use relevant information from the provided evidence
- Keep logic clear with step-by-step reasoning
- Output ONLY the chain of thought in English
- Do NOT include the reference answer directly - synthesize your own answer based on evidence

Please begin your analysis:
"""
        return prompt

    def _fallback_cot(self, question: str, evidence: str, category: str, reference_answer: str) -> str:
        """API 失败时的备用 CoT 生成"""
        if not evidence:
            return f"For the {category} task, I need to analyze the video to answer: {question}. Based on the video content, {reference_answer if reference_answer else 'I would provide a detailed answer after observation.'}."

        return f"""For the {category} task, I need to analyze the video evidence:

{evidence}

Based on these observations, {reference_answer if reference_answer else 'I can derive the answer from the video evidence.'}."""


def load_json(file_path: str) -> Any:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str) -> None:
    """保存 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_question_annotation(video_id: str, question_id: str, annotation_root: str) -> Optional[Dict]:
    """获取问题的标注信息"""
    # 尝试 questions_new
    question_file = Path(annotation_root) / 'questions_new' / video_id / f'{video_id}_questions.json'

    if not question_file.exists():
        # 回退到 questions
        question_file = Path(annotation_root) / 'questions' / video_id / f'{video_id}_questions.json'

    if not question_file.exists():
        return None

    try:
        with open(question_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for q in data.get('questions', []):
            if q.get('question_id') == question_id:
                return q
    except Exception as e:
        print(f"Warning: Could not read {question_file}: {e}")

    return None


def extract_video_id(video_path: str) -> str:
    """从视频路径提取 video_id"""
    # 路径格式: /media/nas_data/zxp_data/output/metadata/task15/task15.mp4
    path = Path(video_path)
    # 尝试从父目录名获取
    if path.parent.name:
        return path.parent.name
    # 从文件名获取
    if path.stem:
        return path.stem
    return "unknown"


def regenerate_nocot_data(
    input_file: str,
    output_file: str,
    annotation_root: str
):
    """重新生成 Non-CoT 数据，使用正确的答案"""
    print(f"读取数据: {input_file}")
    data = load_json(input_file)

    regenerated = []
    total = len(data)
    fixed_count = 0
    short_answer_count = 0

    for i, item in enumerate(data):
        if (i + 1) % 100 == 0:
            print(f"进度: {i + 1}/{total}")

        video_path = item.get('videos', [None])[0]
        video_id = extract_video_id(video_path) if video_path else "unknown"

        question_id = item.get('_meta', {}).get('question_id', '')
        question_type = item.get('_meta', {}).get('question_type', '')

        # 获取问题的标注信息
        annotation = get_question_annotation(video_id, question_id, annotation_root)

        conversations = item.get('conversations', [])
        for conv in conversations:
            if conv.get('from') != 'gpt':
                continue

            current_answer = conv.get('value', '')

            # 对于 short_answer 类型，使用 reference_answer
            if question_type == 'short_answer':
                short_answer_count += 1
                if annotation:
                    reference_answer = annotation.get('reference_answer', '')
                    if reference_answer and current_answer != reference_answer:
                        conv['value'] = reference_answer
                        fixed_count += 1
                else:
                    # 没有找到标注，尝试保持原答案或使用占位符
                    if not current_answer or current_answer.strip() in ['', '...', '…']:
                        conv['value'] = "Please analyze the video to provide a comprehensive answer about the character's eating behavior."

        regenerated.append(item)

    print(f"\n统计:")
    print(f"  总条目: {total}")
    print(f"  Short answer 题目: {short_answer_count}")
    print(f"  修复/更新数量: {fixed_count}")

    # 保存
    save_json(regenerated, output_file)
    print(f"Non-CoT 数据已保存到: {output_file}")


def regenerate_cot_data(
    input_file: str,
    output_file: str,
    annotation_root: str,
    use_api: bool = True,
    model: str = "qwen3.6-flash",
    enable_thinking: bool = False
):
    """重新生成 CoT 数据，根据 answer_source 提取证据并重写思维链"""
    print(f"读取数据: {input_file}")
    data = load_json(input_file)

    extractor = EvidenceExtractor(annotation_root)
    rewriter = CoTReWriter(model=model, enable_thinking=enable_thinking) if use_api else None

    regenerated = []
    total = len(data)
    rewrite_count = 0
    skip_count = 0
    error_count = 0

    for i, item in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"进度: {i + 1}/{total}, 重写: {rewrite_count}, 跳过: {skip_count}, 错误: {error_count}")

        video_path = item.get('videos', [None])[0]
        video_id = extract_video_id(video_path) if video_path else "unknown"

        question_id = item.get('_meta', {}).get('question_id', '')
        annotation = get_question_annotation(video_id, question_id, annotation_root)

        conversations = item.get('conversations', [])
        user_question = ""
        for conv in conversations:
            if conv.get('from') == 'human':
                user_question = conv.get('value', '').replace('<video>\n', '')
                break

        for conv in conversations:
            if conv.get('from') != 'gpt':
                continue

            if annotation:
                category = annotation.get('category', '')
                difficulty = annotation.get('difficulty', '')
                answer_source = annotation.get('answer_source', {})
                reference_answer = annotation.get('reference_answer', '')

                # 提取证据
                evidence = extractor.extract_evidence(answer_source, category, video_id)

                if use_api and rewriter and user_question:
                    try:
                        new_cot = rewriter.rewrite_reasoning(
                            question=user_question,
                            evidence=evidence,
                            category=category,
                            difficulty=difficulty,
                            reference_answer=reference_answer
                        )

                        # 添加回原来的答案（如果不是 short_answer）
                        if annotation.get('question_type') == 'short_answer':
                            final_answer = reference_answer or "Based on the video analysis..."
                        else:
                            final_answer = annotation.get('answer', '') or reference_answer

                        # 使用 think 标签格式
                        if final_answer:
                            conv['value'] = f"</think>\n{new_cot}\n\nAnswer: {final_answer}\n"
                        else:
                            conv['value'] = f"</think>\n{new_cot}\n"

                        rewrite_count += 1
                    except Exception as e:
                        print(f"警告: API调用失败 ({question_id}): {e}")
                        error_count += 1
                        # 使用简化的 CoT
                        if evidence:
                            answer = annotation.get('answer', '')
                            if annotation.get('question_type') == 'short_answer':
                                final_ans = reference_answer or answer
                            else:
                                final_ans = answer

                            reasoning = f"For the {category} task, I need to analyze the video evidence:\n{evidence}\n\nBased on these observations,"
                            if final_ans:
                                conv['value'] = f"</think>\n{reasoning} I can conclude that the answer is: {final_ans}\n"
                            else:
                                conv['value'] = f"</think>\n{reasoning} I can derive the answer from the evidence.\n"
                else:
                    # 不使用 API，使用简化的格式
                    if evidence:
                        # 获取答案
                        answer = annotation.get('answer', '')
                        # 对于选择题，答案是选项字母；对于 short_answer，用 reference_answer
                        if annotation.get('question_type') == 'short_answer':
                            final_ans = reference_answer or answer
                        else:
                            final_ans = answer

                        # 构建思维过程
                        reasoning = f"For the {category} task, I need to analyze the video evidence:\n{evidence}\n\nBased on these observations,"
                        if final_ans:
                            conv['value'] = f"</think>\n{reasoning} I can conclude that the answer is: {final_ans}\n\n"
                        else:
                            conv['value'] = f"</think>\n{reasoning} I can derive the answer from the evidence.\n\n"
                        rewrite_count += 1
                    else:
                        # 证据为空，保持原样
                        conv['value'] = conv.get('value', '')
                        rewrite_count += 1
            else:
                # 没有找到标注，尝试从 _meta 获取 category 并处理
                category = item.get('_meta', {}).get('category', '')
                question_type = item.get('_meta', {}).get('question_type', '')
                original_value = conv.get('value', '')

                # 尝试根据 category 推断 answer_source 路径
                if category:
                    # 构建可能的文件路径
                    module_map = {
                        'action_timeline': 'action_timeline',
                        'bite': 'bite',
                        'chewing': 'chewing',
                        'composition': 'composition',
                        'pace': 'pace',
                        'intake': 'intake',
                        'facial_expression': 'facial_expression_sequence',
                        'behavioral_health': 'behavioral_health_analysis',
                        'metadata': 'metadata'
                    }
                    module_name = module_map.get(category, category)

                    # 尝试读取文件
                    file_path = f"{module_name}/{video_id}/{video_id}_{module_name}.json"
                    full_path = Path(annotation_root) / file_path

                    if full_path.exists():
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                evidence_data = json.load(f)
                            evidence = extractor._format_evidence(evidence_data, category, {})

                            # 提取答案（从原始 CoT 中提取）
                            final_ans = ""
                            if '</think>\n' in original_value:
                                parts = original_value.split('</think>\n')
                                if len(parts) > 1:
                                    # 取最后一部分作为答案
                                    final_ans = parts[-1].strip()
                            elif question_type == 'short_answer':
                                # short_answer 通常没有 think 标签，保留原文
                                conv['value'] = original_value
                                rewrite_count += 1
                                continue

                            reasoning = f"For the {category} task, I need to analyze the video evidence:\n{evidence}\n\nBased on these observations,"
                            if final_ans:
                                conv['value'] = f"</think>\n{reasoning} I can conclude that the answer is: {final_ans}\n\n"
                            else:
                                conv['value'] = f"</think>\n{reasoning} I can derive the answer from the evidence.\n\n"
                            rewrite_count += 1
                        except Exception as e:
                            print(f"Warning: Could not process {file_path}: {e}")
                            # 保持原样
                            conv['value'] = original_value
                            rewrite_count += 1
                    else:
                        # 文件不存在，保持原样
                        conv['value'] = original_value
                        rewrite_count += 1
                else:
                    # 没有 category，保持原样
                    conv['value'] = original_value
                    rewrite_count += 1

        regenerated.append(item)

    print(f"\n统计:")
    print(f"  总条目: {total}")
    print(f"  重写成功: {rewrite_count}")
    print(f"  跳过: {skip_count}")
    print(f"  错误: {error_count}")

    # 保存
    save_json(regenerated, output_file)
    print(f"CoT 数据已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='训练数据重新生成脚本')
    parser.add_argument('--mode', type=str, required=True, choices=['cot', 'no_cot'],
                       help='生成模式: cot=生成思维链版本, no_cot=生成无思维链版本')
    parser.add_argument('--input', type=str,
                       default=None,
                       help='输入数据文件路径（默认使用现有数据文件）')
    parser.add_argument('--output', type=str,
                       default=None,
                       help='输出数据文件路径')
    parser.add_argument('--annotation_root', type=str,
                       default=DEFAULT_ANNOTATION_ROOT,
                       help='标注数据根目录')
    parser.add_argument('--use_api', action='store_true',
                       help='使用 qwen API 重写思维链（仅 cot 模式）')
    parser.add_argument('--model', type=str, default='qwen3.6-flash',
                       help='qwen 模型名称 (默认: qwen3.6-flash, 可选: qwen-turbo, qwen-max, qwen-long)')
    parser.add_argument('--enable_thinking', action='store_true',
                       help='启用深度思考模式（仅 cot + use_api 模式）')

    args = parser.parse_args()

    # 设置默认路径
    if args.mode == 'cot':
        input_file = args.input or DEFAULT_COT_INPUT
        output_file = args.output or DEFAULT_COT_OUTPUT
    else:
        input_file = args.input or DEFAULT_NOCOT_INPUT
        output_file = args.output or DEFAULT_NOCOT_OUTPUT

    print(f"模式: {args.mode}")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")
    print(f"标注根目录: {args.annotation_root}")
    if args.mode == 'cot':
        print(f"使用API: {args.use_api}")
        if args.use_api:
            print(f"模型: {args.model}")
            print(f"深度思考: {args.enable_thinking}")
    print()

    if args.mode == 'cot':
        regenerate_cot_data(
            input_file=input_file,
            output_file=output_file,
            annotation_root=args.annotation_root,
            use_api=args.use_api,
            model=args.model,
            enable_thinking=args.enable_thinking
        )
    else:
        regenerate_nocot_data(
            input_file=input_file,
            output_file=output_file,
            annotation_root=args.annotation_root
        )

    print()
    print("完成!")


if __name__ == '__main__':
    main()
