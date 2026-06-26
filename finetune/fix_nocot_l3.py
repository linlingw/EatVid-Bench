#!/usr/bin/env python3
"""
修复 Non-CoT 数据中 L3 问题的空答案
从标注数据中提取 reference_answer
"""

import json
from pathlib import Path
from typing import Optional


def load_json(file_path: str) -> dict:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_question_annotation(video_id: str, question_id: str, output_root: str) -> Optional[dict]:
    """从标注数据中获取问题的标注信息

    Args:
        video_id: 视频 ID，如 task15
        question_id: 问题 ID，如 task15_short_001
        output_root: 标注数据根目录

    Returns:
        问题标注字典，如果未找到则返回 None
    """
    # 先尝试 questions_new 文件夹
    question_file = Path(output_root) / 'questions_new' / video_id / f'{video_id}_questions.json'

    # 如果不存在，尝试 questions 文件夹
    if not question_file.exists():
        question_file = Path(output_root) / 'questions' / video_id / f'{video_id}_questions.json'

    if not question_file.exists():
        return None

    try:
        data = load_json(str(question_file))
        for q in data.get('questions', []):
            if q.get('question_id') == question_id:
                return q
    except Exception as e:
        print(f"Warning: Could not read {question_file}: {e}")

    return None


def is_empty_or_placeholder_answer(value: str) -> bool:
    """检查答案是否为空或占位符（如...）"""
    if not value:
        return True

    # 去除换行符和空白字符
    cleaned = value.replace('\n', '').replace('\r', '').strip()

    # 检查是否为空或只有占位符
    # 可能的占位符: "...", "…" (U+2026), "。"
    if cleaned == '':
        return True

    # 检查 ASCII 省略号
    if cleaned == '...':
        return True

    # 检查 Unicode 省略号 (U+2026)
    if cleaned == '…':
        return True

    # 检查去除省略号后是否还有其他内容
    test = cleaned.replace('...', '').replace('…', '').replace('。', '').strip()
    if test == '':
        return True

    return False


def fix_nocot_l3_answers(
    nocot_file: str,
    output_file: str,
    output_root: str = '/media/nas_data/zxp_data/output'
):
    """修复 Non-CoT 数据中 L3 问题的空答案

    从标注数据中提取 reference_answer 填充到 nocot 数据中
    """
    print(f"读取 Non-CoT 数据: {nocot_file}")
    nocot_data = load_json(nocot_file)

    fixed_count = 0
    empty_count = 0
    not_found_count = 0
    no_ref_answer_count = 0

    for item in nocot_data:
        # 获取 video_id 和 question_id
        videos = item.get('videos', [])
        if not videos:
            continue

        video_path = videos[0]
        # 从路径中提取 video_id，如 /media/nas_data/zxp_data/output/metadata/task15/task15.mp4
        video_id = Path(video_path).parent.name

        question_id = item.get('_meta', {}).get('question_id', '')
        question_type = item.get('_meta', {}).get('question_type', '')

        # 只处理 short_answer 类型（L3）
        if question_type != 'short_answer':
            continue

        for conv in item.get('conversations', []):
            if conv.get('from') != 'gpt':
                continue

            current_value = conv.get('value', '')

            # 检查是否是空答案或占位符
            if is_empty_or_placeholder_answer(current_value):
                empty_count += 1

                # 从标注数据获取参考答案
                annotation = get_question_annotation(video_id, question_id, output_root)

                if annotation:
                    reference_answer = annotation.get('reference_answer', '')
                    if reference_answer:
                        conv['value'] = reference_answer
                        fixed_count += 1
                        print(f"Fixed: {question_id} (video: {video_id})")
                    else:
                        no_ref_answer_count += 1
                        print(f"Warning: {question_id} has no reference_answer")
                else:
                    not_found_count += 1
                    print(f"Warning: Annotation not found for {question_id}")

    print(f"\n统计:")
    print(f"  空答案数量: {empty_count}")
    print(f"  修复数量: {fixed_count}")
    print(f"  标注未找到: {not_found_count}")
    print(f"  无参考答案: {no_ref_answer_count}")

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nocot_data, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到: {output_file}")


def main():
    nocot_file = '/media/nas_data/zxp_data/LlamaFactory/data/train_data_no_cot.json'
    output_file = '/media/nas_data/zxp_data/LlamaFactory/data/train_data_no_cot_fixed.json'
    output_root = '/media/nas_data/zxp_data/output'

    fix_nocot_l3_answers(nocot_file, output_file, output_root)
    print("\n完成!")


if __name__ == '__main__':
    main()
