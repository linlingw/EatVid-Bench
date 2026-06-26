"""
AGCoT 推理链生成脚本

从 output/ 标注目录读取各模块结果，为每个 task_id 生成
自然语言推理链（Chain-of-Thought），供训练数据构造使用。

Usage:
    # 为单个任务生成 CoT（调试）
    python generate_cot.py --annotation_root /path/to/output \
        --task_id task34 --print

    # 为 task_ids_file 中的所有任务批量生成 CoT，保存到 output_dir
    python generate_cot.py --annotation_root /path/to/output \
        --task_ids_file worst_tasks_train.json \
        --output_dir finetune/cot_cache
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────── annotation loaders ───────────────────────────────

def _load(annotation_root: Path, module: str, task_id: str) -> Optional[Dict]:
    p = annotation_root / module / task_id / f"{task_id}_{module}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
    return None


# ─────────────────────────── CoT builders per module ──────────────────────────

def _cot_bite(d: Dict) -> str:
    n = d.get("num_bites", 0)
    events = d.get("bite_events", [])
    hands = [e.get("hand", "Unknown") for e in events]
    right = hands.count("Right")
    left = hands.count("Left")
    dur = d.get("duration", 0)
    lines = [
        f"[Bite Analysis] Total bites detected: {n} over {dur:.1f}s.",
        f"  Hand usage — Right: {right}, Left: {left}.",
    ]
    if events:
        ts = [e.get("timestamp", 0) for e in events]
        lines.append(f"  First bite at {ts[0]:.1f}s, last at {ts[-1]:.1f}s.")
        if len(ts) > 1:
            intervals = [ts[i+1]-ts[i] for i in range(len(ts)-1)]
            avg_iv = sum(intervals)/len(intervals)
            lines.append(f"  Average inter-bite interval: {avg_iv:.1f}s.")
    return "\n".join(lines)


def _cot_composition(d: Dict) -> str:
    comp = (d.get("composition") or {}).get("composition_by_count", {})
    if not comp:
        return "[Food Composition] No composition data available."
    items = sorted(comp.items(), key=lambda x: x[1].get("count", 0), reverse=True)
    desc = ", ".join(f"{k} ({v.get('percentage', 0):.1f}%)" for k, v in items if v.get("count", 0) > 0)
    dom = items[0][0] if items else "unknown"
    return f"[Food Composition] Dominant food type: {dom}. Distribution: {desc}."


def _cot_pace(d: Dict) -> str:
    overall = d.get("overall_rate", {})
    bpm = overall.get("bites_per_minute", 0)
    avg_iv = overall.get("avg_interval_seconds", 0)
    std_iv = overall.get("std_interval_seconds", 0)
    windows = d.get("temporal_windows", [])
    pace_label = "fast" if bpm > 10 else ("slow" if bpm < 5 else "moderate")
    lines = [
        f"[Eating Pace] Overall rate: {bpm:.1f} bites/min → classified as {pace_label}.",
        f"  Avg interval: {avg_iv:.1f}s ± {std_iv:.1f}s.",
    ]
    if windows:
        w_rates = [w.get("bites_per_minute", 0) for w in windows]
        lines.append(f"  Temporal variation: min {min(w_rates):.1f}, max {max(w_rates):.1f} bites/min across {len(windows)} windows.")
    return "\n".join(lines)


def _cot_action(d: Dict) -> str:
    events = d.get("action_timeline", [])
    if not events:
        return "[Action Timeline] No action events recorded."
    key = events[:5]
    desc = "; ".join(f"{e.get('start_time','?')}-{e.get('end_time','?')}: {e.get('event','')}" for e in key)
    return f"[Action Timeline] {len(events)} events total. Key events: {desc}{'...' if len(events)>5 else ''}."


def _cot_expression(d: Dict) -> str:
    segs = d.get("facial_expression_timeline", [])
    if not segs:
        return "[Facial Expression] No expression data."
    desc = "; ".join(f"{s.get('start_time','?')}: {s.get('description','')}" for s in segs[:4])
    return f"[Facial Expression] {len(segs)} segments. Sample: {desc}."


def _cot_behavioral(d: Dict) -> str:
    bha = d.get("behavioral_health_analysis", {})
    speed = bha.get("eating_speed", {})
    food_q = bha.get("food_choice_quality", {})
    emotion = bha.get("emotional_eating_intensity", {})
    psych = bha.get("psychological_health_indicators", {})
    lines = [
        f"[Behavioral Health] Eating speed: {speed.get('level','?')} (confidence: {speed.get('confidence','?')}).",
        f"  Food quality: {food_q.get('level','?')}. Emotional eating: {emotion.get('level','?')}.",
        f"  Stress level: {psych.get('stress_level','?')}, Self-control: {psych.get('self_control','?')}.",
    ]
    assess = speed.get("assessment", "")
    if assess:
        lines.append(f"  Assessment: {assess}")
    return "\n".join(lines)


def _cot_chewing(d: Dict) -> str:
    total = d.get("total_chews", 0)
    dur = d.get("duration", 0)
    events = d.get("chewing_events", [])
    avg_conf = sum(e.get("confidence", 0) for e in events) / len(events) if events else 0
    return (f"[Chewing] Total chewing events: {total} over {dur:.1f}s. "
            f"Avg detection confidence: {avg_conf:.2f}.")


def _cot_intake(d: Dict) -> str:
    ca = d.get("comparison_analysis") or {}
    foods = ca.get("consumption_by_food", [])
    if not foods:
        return "[Intake] No intake estimation available."
    parts = []
    for f in foods:
        ft = f.get("food_type", "?")
        ratio = f.get("consumption_ratio", 0)
        parts.append(f"{ft}: {ratio*100:.1f}% consumed")
    return f"[Intake Estimation] {'; '.join(parts)}."


# ─────────────────────────── main CoT generator ───────────────────────────────

def generate_cot(annotation_root: Path, task_id: str) -> str:
    """Generate full AGCoT reasoning chain for one task."""
    root = Path(annotation_root)
    blocks = []

    bite = _load(root, "bite", task_id)
    if bite:
        blocks.append(_cot_bite(bite))

    comp = _load(root, "composition", task_id)
    if comp:
        blocks.append(_cot_composition(comp))

    pace = _load(root, "pace", task_id)
    if pace:
        blocks.append(_cot_pace(pace))

    action = _load(root, "action_timeline", task_id)
    if action:
        blocks.append(_cot_action(action))

    expr = _load(root, "facial_expression_sequence", task_id)
    if expr:
        blocks.append(_cot_expression(expr))

    chew = _load(root, "chewing", task_id)
    if chew:
        blocks.append(_cot_chewing(chew))

    intake = _load(root, "intake", task_id)
    if intake:
        blocks.append(_cot_intake(intake))

    bha = _load(root, "behavioral_health_analysis", task_id)
    if bha:
        blocks.append(_cot_behavioral(bha))

    if not blocks:
        return f"[No annotation data found for {task_id}]"
    return "\n".join(blocks)


# ─────────────────────────── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate AGCoT reasoning chains from annotations")
    parser.add_argument("--annotation_root", required=True, help="Path to output/ annotation directory")
    parser.add_argument("--task_id", default=None, help="Single task ID (debug mode)")
    parser.add_argument("--task_ids_file", default=None, help="JSON file with worst_task_ids list")
    parser.add_argument("--output_dir", default=None, help="Directory to save per-task CoT JSON files")
    parser.add_argument("--print", action="store_true", dest="do_print", help="Print CoT to stdout")
    args = parser.parse_args()

    root = Path(args.annotation_root)
    task_ids = []

    if args.task_id:
        task_ids = [args.task_id]
    elif args.task_ids_file:
        data = json.loads(Path(args.task_ids_file).read_text(encoding="utf-8"))
        task_ids = data.get("worst_task_ids", data) if isinstance(data, dict) else data
    else:
        parser.error("Provide --task_id or --task_ids_file")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for tid in task_ids:
        cot = generate_cot(root, tid)
        results[tid] = cot
        if args.do_print:
            print(f"\n{'='*60}\n{tid}\n{'='*60}\n{cot}")
        if out_dir:
            (out_dir / f"{tid}_cot.json").write_text(
                json.dumps({"task_id": tid, "cot": cot}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

    if out_dir and not args.do_print:
        logger.info(f"Saved {len(results)} CoT files to {out_dir}")
    elif not args.do_print:
        # Save combined file when no output_dir specified
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

