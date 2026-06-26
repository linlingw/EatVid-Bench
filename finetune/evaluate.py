"""
AGCoT-LoRA 微调模型评估脚本

在测试集（hard subset，30 个 task_ids）上运行微调后的模型，
使用与 benchmark 完全相同的评估逻辑，输出各维度指标和对比报告。

Usage:
    python evaluate.py \
        --task_ids_file worst_tasks_test.json \
        --annotation_root /path/to/output \
        --video_root /path/to/videos \
        --model_name_or_path /path/to/finetuned_checkpoint \
        --output_dir finetune/eval_results \
        [--baseline_results ../../results/true_output]  # 对比基线结果
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────── data loading ────────────────────────────────────

def load_task_ids(path: str) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("worst_task_ids", data) if isinstance(data, dict) else data


def load_questions(annotation_root: Path, task_id: str) -> List[Dict]:
    p = annotation_root / "questions" / task_id / f"{task_id}_questions.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("questions", []) if isinstance(d, dict) else d


def find_video(annotation_root: Path, task_id: str) -> Optional[str]:
    """从 annotation_root/metadata/{task_id}/{task_id}.mp4 定位视频。"""
    for ext in ("mp4", "avi", "mov", "mkv"):
        p = annotation_root / "metadata" / task_id / f"{task_id}.{ext}"
        if p.exists():
            return str(p.resolve())
    return None


# ──────────────────────────── model inference ─────────────────────────────────

def load_model(model_path: str):
    """Load fine-tuned model via transformers (Qwen2-VL style)."""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch
    logger.info(f"Loading model from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def run_inference(model, processor, question: Dict, video_path: Optional[str]) -> str:
    """Run inference for a single question. Returns raw model output string."""
    import torch

    q_text = question.get("question", "")
    options = question.get("options", [])
    if options:
        q_text += "\n" + "\n".join(options)

    if video_path:
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
            {"type": "text", "text": q_text},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
    else:
        messages = [{"role": "user", "content": q_text}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ──────────────────────────── scoring ─────────────────────────────────────────

def score_prediction(q: Dict, prediction: str) -> Dict:
    """Simple rule-based scoring matching qa_metrics.py logic."""
    qt = q.get("question_type", "")
    gt = str(q.get("answer") or q.get("reference_answer") or "").strip()

    pred_clean = prediction.strip().upper()
    gt_clean = gt.strip().upper()

    # Extract first letter for choice questions
    import re
    first_letter = re.search(r"^([A-D])", pred_clean)

    if qt in ("single_choice", "true_false"):
        pred_key = first_letter.group(1) if first_letter else pred_clean[:1]
        correct = pred_key == gt_clean[:1]
        score = 1.0 if correct else 0.0
    elif qt == "multi_choice":
        pred_letters = set(re.findall(r"[A-D]", pred_clean))
        gt_letters = set(re.findall(r"[A-D]", gt_clean))
        correct = pred_letters == gt_letters
        score = len(pred_letters & gt_letters) / max(len(gt_letters), 1) if gt_letters else 0.0
    elif qt == "fill_blank":
        tol = q.get("answer_tolerance")
        try:
            pred_num = float(re.search(r"[\d.]+", pred_clean).group())
            gt_num = float(re.search(r"[\d.]+", gt_clean).group())
            tol_val = float(tol) if tol else abs(gt_num * 0.1)
            correct = abs(pred_num - gt_num) <= tol_val
            score = 1.0 if correct else 0.0
        except Exception:
            correct = pred_clean == gt_clean
            score = 1.0 if correct else 0.0
    else:  # short_answer
        correct = False
        score = 0.0  # requires GPT judge; placeholder

    return {"correct": correct, "score": score, "prediction_clean": pred_clean, "ground_truth_clean": gt_clean}


def compute_metrics(records: List[Dict]) -> Dict:
    """Compute weighted accuracy metrics matching qa_metrics.py."""
    diff_w = {"L1": 1.0, "L2": 2.0, "L3": 3.0}
    cat_w = {"action_timeline": 1.0, "bite": 1.0, "facial_expression": 1.0,
             "composition": 1.0, "pace": 1.0, "metadata": 0.5, "behavioral_health": 2.0}
    type_base = {"single_choice": 1.0, "multi_choice": 2.0, "true_false": 0.5,
                 "fill_blank": 1.5, "short_answer": 5.0}

    total, correct_n = len(records), 0
    w_score = w_max = 0.0
    by_type: Dict[str, Dict] = {}

    for r in records:
        qt = r.get("question_type", "")
        ms = type_base.get(qt, 1.0) * diff_w.get(r.get("difficulty", "L1"), 1.0) * cat_w.get(r.get("category", ""), 1.0)
        c = r.get("correct", False)
        s = r.get("score", 0.0)
        correct_n += int(c)
        w_score += s * ms
        w_max += ms
        if qt not in by_type:
            by_type[qt] = {"total": 0, "correct": 0, "w_score": 0.0, "w_max": 0.0}
        by_type[qt]["total"] += 1
        by_type[qt]["correct"] += int(c)
        by_type[qt]["w_score"] += s * ms
        by_type[qt]["w_max"] += ms

    return {
        "total_questions": total,
        "correct_questions": correct_n,
        "accuracy": correct_n / total if total else 0.0,
        "weighted_accuracy": w_score / w_max if w_max else 0.0,
        "by_type": {qt: {
            "total": v["total"], "correct": v["correct"],
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
            "weighted_accuracy": v["w_score"] / v["w_max"] if v["w_max"] else 0.0,
        } for qt, v in by_type.items()},
    }


# ──────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate AGCoT fine-tuned model on hard subset")
    parser.add_argument("--task_ids_file", required=True)
    parser.add_argument("--annotation_root", required=True,
                        help="Path to output/ dir (contains metadata/, questions/, etc.)")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", default="finetune/eval_results")
    parser.add_argument("--baseline_results", default=None,
                        help="Path to baseline model result dir for comparison table")
    args = parser.parse_args()

    task_ids = load_task_ids(args.task_ids_file)
    annotation_root = Path(args.annotation_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating on {len(task_ids)} tasks")
    model, processor = load_model(args.model_name_or_path)

    records = []
    for task_id in task_ids:
        questions = load_questions(annotation_root, task_id)
        video_path = find_video(annotation_root, task_id)

        for q in questions:
            t0 = time.time()
            try:
                pred = run_inference(model, processor, q, video_path)
            except Exception as e:
                logger.warning(f"Inference error {q.get('question_id')}: {e}")
                pred = ""
            eval_d = score_prediction(q, pred)
            records.append({
                "question_id": q.get("question_id"),
                "video_id": task_id,
                "question_type": q.get("question_type"),
                "difficulty": q.get("difficulty"),
                "category": q.get("category"),
                "question": q.get("question"),
                "prediction": pred,
                "ground_truth": q.get("answer") or q.get("reference_answer"),
                **eval_d,
                "latency_s": round(time.time() - t0, 2),
            })

    # Save per-question results
    results_path = out_dir / "qa_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = compute_metrics(records)
    metrics_path = out_dir / "qa_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Results: accuracy={metrics['accuracy']:.4f}, weighted_accuracy={metrics['weighted_accuracy']:.4f}")
    logger.info(f"Saved to {out_dir}")

    # Optional: comparison table vs baseline
    if args.baseline_results:
        bl_path = Path(args.baseline_results) / "qa_evaluation" / "all" / "qa_metrics.json"
        if bl_path.exists():
            bl = json.loads(bl_path.read_text(encoding="utf-8"))
            bl_metrics = bl.get("metrics", bl)
            print("\n=== Comparison: Baseline vs AGCoT-LoRA ===")
            print(f"{'Metric':<30} {'Baseline':>10} {'AGCoT-LoRA':>12}")
            print("-" * 55)
            for k in ("accuracy", "weighted_accuracy"):
                bl_v = bl_metrics.get(k, 0.0)
                ft_v = metrics.get(k, 0.0)
                print(f"{k:<30} {bl_v:>10.4f} {ft_v:>12.4f}  ({ft_v-bl_v:+.4f})")


if __name__ == "__main__":
    main()

