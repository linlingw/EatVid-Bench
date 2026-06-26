"""
AGCoT 训练数据准备脚本

读取筛选出的 task_ids（训练集），从标注目录生成 CoT 推理链，
结合每个视频的 QA 题，输出 LLaMA-Factory ShareGPT 格式 JSONL。

输出格式（LLaMA-Factory video ShareGPT）：
  [
    {
      "conversations": [
        {"from": "human", "value": "<video>\n{question_with_options}"},
        {"from": "gpt",   "value": "<think>\n{cot}\n</think>\n{answer}"}
      ],
      "videos": ["/abs/path/to/task_id.mp4"]
    },
    ...
  ]

Usage:
    python prepare_train_data.py \
        --task_ids_file worst_tasks_train.json \
        --annotation_root /path/to/output \
        --video_root /path/to/videos \
        --output_file finetune/train_data.json \
        [--exclude_types short_answer]   # 可选：排除某些题型
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 允许直接从 experiments/ 运行
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from finetune.generate_cot import generate_cot


# ──────────────────────────── helpers ─────────────────────────────────────────

def load_task_ids(path: str) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("worst_task_ids", [])
    return data  # plain list


def load_questions(annotation_root: Path, task_id: str) -> List[Dict]:
    """Load pre-generated questions from output/questions/{task_id}/."""
    p = annotation_root / "questions" / task_id / f"{task_id}_questions.json"
    if not p.exists():
        logger.warning(f"No questions file: {p}")
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("questions", []) if isinstance(d, dict) else d


def find_video(annotation_root: Path, task_id: str) -> Optional[str]:
    """从 annotation_root/metadata/{task_id}/{task_id}.mp4 定位视频。"""
    for ext in ("mp4", "avi", "mov", "mkv"):
        p = annotation_root / "metadata" / task_id / f"{task_id}.{ext}"
        if p.exists():
            return str(p.resolve())
    logger.debug(f"Video not found for {task_id} under {annotation_root}/metadata/")
    return None


def format_question_text(q: Dict) -> str:
    """Format question + options into a single user-turn string."""
    text = q.get("question", "")
    options = q.get("options", [])
    if options:
        text += "\n" + "\n".join(options)
    return text


def format_answer(q: Dict, cot: str, include_cot: bool) -> str:
    """Format the model's expected answer turn."""
    ans = q.get("answer") or q.get("reference_answer") or ""
    if not include_cot or not cot:
        return str(ans)
    rethink = "I have already analyzed quite a bit, but I need to further confirm the accuracy of my analysis based on the actual content of the video"
    return f"<think>\n{cot}\n{rethink}\n</think>\n{ans}"


def build_sample(q: Dict, cot: str, video_path: Optional[str], include_cot: bool) -> Dict:
    human_val = "<video>\n" + format_question_text(q)
    gpt_val = format_answer(q, cot, include_cot)
    sample = {
        "conversations": [
            {"from": "human", "value": human_val},
            {"from": "gpt",   "value": gpt_val},
        ]
    }
    if video_path:
        sample["videos"] = [video_path]
    # Keep metadata for traceability
    sample["_meta"] = {
        "question_id": q.get("question_id"),
        "question_type": q.get("question_type"),
        "difficulty": q.get("difficulty"),
        "category": q.get("category"),
    }
    return sample


# ──────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare LLaMA-Factory training data with AGCoT")
    parser.add_argument("--task_ids_file", required=True,
                        help="JSON file with worst_task_ids (train split)")
    parser.add_argument("--annotation_root", required=True,
                        help="Path to output/ directory (contains metadata/, questions/, bite/, etc.)")
    parser.add_argument("--output_file", default="finetune/train_data.json",
                        help="Output JSON file path (LLaMA-Factory format)")
    parser.add_argument("--exclude_types", default="",
                        help="Comma-separated question types to exclude (e.g. short_answer)")
    parser.add_argument("--no_cot", action="store_true",
                        help="Omit CoT from answers (ablation baseline)")
    args = parser.parse_args()

    task_ids = load_task_ids(args.task_ids_file)
    annotation_root = Path(args.annotation_root)
    exclude_types = {t.strip() for t in args.exclude_types.split(",") if t.strip()}
    include_cot = not args.no_cot

    logger.info(f"Processing {len(task_ids)} tasks | CoT={'yes' if include_cot else 'no'} | exclude_types={exclude_types or 'none'}")

    samples = []
    skipped_no_video = 0
    skipped_no_questions = 0

    for task_id in task_ids:
        questions = load_questions(annotation_root, task_id)
        if not questions:
            skipped_no_questions += 1
            continue

        cot = generate_cot(annotation_root, task_id) if include_cot else ""
        # 视频路径从 annotation_root/metadata/{task_id}/{task_id}.mp4 自动推断
        video_path = find_video(annotation_root, task_id)
        if video_path is None:
            skipped_no_video += 1

        for q in questions:
            if q.get("question_type") in exclude_types:
                continue
            sample = build_sample(q, cot, video_path, include_cot)
            samples.append(sample)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Done. {len(samples)} training samples saved to {out_path}")
    if skipped_no_questions:
        logger.warning(f"  {skipped_no_questions} tasks had no question file (skipped)")
    if skipped_no_video:
        logger.warning(f"  {skipped_no_video} tasks had no video file (samples kept, no video path)")


if __name__ == "__main__":
    main()

