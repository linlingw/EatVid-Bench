"""
GPT-based Judge for Short Answer QA Evaluation

This module provides GPT-based evaluation for short-answer questions,
supporting both real-time evaluation during benchmark runs and
backfill mode for post-processing existing results.

Usage:
    # During evaluation (enabled via config)
    config = QAScoreConfig(enable_gpt4_judge=True, gpt4_api_key="...")
    metrics = QAMetrics(config)
    score = metrics.evaluate_short_answer(...)

    # Backfill mode (post-processing)
    from eatvid_benchmark.utils.gpt_judge import batch_backfill_gpt_scores
    batch_backfill_gpt_scores(results_dir, api_key="...")
"""

import logging
import os
import re
import json
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


# Default API configuration
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


# ─────────────────────────── Prompt Templates ────────────────────────────

BATCH_JUDGE_SYSTEM = (
    "You are an expert evaluator for fine-grained eating behavior video analysis. "
    "Score model answers against reference answers strictly and objectively."
)

BATCH_JUDGE_PROMPT = """You will evaluate {n} model answers for eating behavior video analysis questions.
For each item, compare the model answer to the reference answer and assign a score from 0.0 to 1.0:
  1.0 = fully correct and complete
  0.7~0.9 = mostly correct with minor omissions
  0.4~0.6 = partially correct
  0.1~0.3 = mostly wrong but contains some relevant content
  0.0 = completely wrong or irrelevant

Output ONLY a JSON array of {n} numbers in the same order, e.g.: [0.8, 0.3, 1.0, ...]
Do NOT include any explanation or extra text.

Items to evaluate:
{items}"""

ITEM_TEMPLATE = """--- Item {idx} ---
Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}"""

SINGLE_JUDGE_PROMPT = """Evaluate the following model answer for a video-based eating behavior analysis question.

Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}

Assign a score from 0.0 to 1.0 based on accuracy and completeness:
  1.0 = fully correct and complete
  0.7~0.9 = mostly correct with minor omissions
  0.4~0.6 = partially correct
  0.1~0.3 = mostly wrong but contains some relevant content
  0.0 = completely wrong or irrelevant

Output ONLY a single number (the score), with no explanation."""


# ─────────────────────────── GPT Judge Client ───────────────────────────────

@dataclass
class GPTJudgeConfig:
    """Configuration for GPT-based judge."""
    api_key: Optional[str] = None
    base_url: str = DEFAULT_API_BASE
    model: str = DEFAULT_MODEL
    batch_size: int = 5
    timeout: int = 30
    max_retries: int = 2


class GPTJudge:
    """
    GPT-based judge for short answer evaluation.

    Supports both single-question scoring and batch processing
    for efficiency.
    """

    def __init__(self, config: Optional[GPTJudgeConfig] = None):
        """
        Initialize GPT judge.

        Args:
            config: Judge configuration. If None, reads from environment:
                    - GPT_JUDGE_API_KEY or OPENAI_API_KEY
                    - GPT_JUDGE_BASE_URL (defaults to OpenAI)
                    - GPT_JUDGE_MODEL (defaults to gpt-4o-mini)
        """
        self.config = config or self._default_config()
        self._client = None

    def _default_config(self) -> GPTJudgeConfig:
        """Create config from environment variables."""
        return GPTJudgeConfig(
            api_key=os.getenv("GPT_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("GPT_JUDGE_BASE_URL", DEFAULT_API_BASE),
            model=os.getenv("GPT_JUDGE_MODEL", DEFAULT_MODEL),
            batch_size=int(os.getenv("GPT_JUDGE_BATCH_SIZE", "5"))
        )

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                if not self.config.api_key:
                    raise ValueError("GPT Judge API key not configured. "
                                   "Set GPT_JUDGE_API_KEY environment variable.")
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                logger.error("openai package not installed. "
                           "Install with: pip install openai")
                raise
        return self._client

    def score_single(
        self,
        question: str,
        reference: str,
        prediction: str
    ) -> Optional[float]:
        """
        Score a single answer using GPT.

        Args:
            question: The question text
            reference: Reference answer
            prediction: Model's predicted answer

        Returns:
            Score from 0.0 to 1.0, or None if scoring fails
        """
        prompt = SINGLE_JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            prediction=prediction
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": BATCH_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=16,
                timeout=self.config.timeout
            )
            raw = resp.choices[0].message.content.strip()
            return self._parse_single_score(raw)
        except Exception as e:
            logger.warning(f"GPT single scoring failed: {e}")
            return None

    def score_batch(
        self,
        items: List[Dict[str, str]]
    ) -> List[Optional[float]]:
        """
        Score multiple answers in batch for efficiency.

        Args:
            items: List of dicts with keys: question, reference, prediction
                   Can also include 'question_id' for tracking

        Returns:
            List of scores (0.0-1.0) in same order as input, None for failures
        """
        scores = []

        for start in range(0, len(items), self.config.batch_size):
            chunk = items[start:start + self.config.batch_size]
            n = len(chunk)

            # Build batch prompt
            items_text = "\n\n".join(
                ITEM_TEMPLATE.format(
                    idx=i + 1,
                    question=it.get("question", ""),
                    reference=it.get("reference", ""),
                    prediction=it.get("prediction", "")
                )
                for i, it in enumerate(chunk)
            )

            prompt = BATCH_JUDGE_PROMPT.format(n=n, items=items_text)

            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": BATCH_JUDGE_SYSTEM},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    timeout=self.config.timeout * n
                )
                raw = resp.choices[0].message.content.strip()
                parsed = self._parse_score_list(raw, n)
            except Exception as e:
                logger.warning(f"GPT batch scoring failed: {e}")
                parsed = [None] * n

            scores.extend(parsed)
            logger.debug(f"Scored {start + n}/{len(items)} items")

        return scores

    def _parse_single_score(self, text: str) -> Optional[float]:
        """Extract a single score from GPT output."""
        # Try to extract a number
        match = re.search(r'0?\.\d+|1\.0|0|1', text)
        if match:
            try:
                val = float(match.group())
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        logger.warning(f"Failed to parse score from: {text!r}")
        return None

    def _parse_score_list(self, text: str, n: int) -> List[Optional[float]]:
        """Extract a list of n scores from GPT output."""
        # Try JSON array format
        m = re.search(r'\[([^\]]+)\]', text)
        if m:
            try:
                nums = [float(x.strip()) for x in m.group(1).split(',')]
                if len(nums) == n:
                    return [max(0.0, min(1.0, v)) for v in nums]
            except ValueError:
                pass

        # Fallback: extract all numbers
        nums = re.findall(r'0?\.\d+|1\.0|0|1', text)
        if len(nums) >= n:
            try:
                return [max(0.0, min(1.0, float(nums[i]))) for i in range(n)]
            except (ValueError, IndexError):
                pass

        logger.warning(f"Failed to parse {n} scores from: {text!r}")
        return [None] * n


# ─────────────────────────── Backfill Mode ────────────────────────────────

def batch_backfill_gpt_scores(
    results_dir: Path,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = 5,
    force: bool = False
) -> Dict[str, Any]:
    """
    Backfill GPT scores for existing evaluation results.

    This function reads qa_results.jsonl, adds GPT scores to short_answer
    questions that don't have them, and recomputes metrics.

    Args:
        results_dir: Path to directory containing qa_evaluation/all/
        api_key: OpenAI API key (or reads from env)
        model: GPT model to use
        batch_size: Questions per API call
        force: Re-score even if gpt4_score exists

    Returns:
        Summary of backfill operation
    """
    api_key = api_key or os.getenv("GPT_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set GPT_JUDGE_API_KEY or pass api_key parameter.")

    config = GPTJudgeConfig(api_key=api_key, model=model, batch_size=batch_size)
    judge = GPTJudge(config)

    # Locate result files
    qa_dir = results_dir / "qa_evaluation" / "all"
    jsonl_path = qa_dir / "qa_results.jsonl"
    metrics_path = qa_dir / "qa_evaluation_full.json"

    if not jsonl_path.exists():
        return {"error": f"Results file not found: {jsonl_path}"}

    # Load results
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter short_answer questions needing scoring
    to_score = []
    for r in records:
        if r.get("question_type") != "short_answer":
            continue
        if not force and (r.get("evaluation_details") or {}).get("gpt4_score") is not None:
            continue
        to_score.append(r)

    if not to_score:
        return {"status": "skip", "reason": "No short_answer questions need scoring"}

    logger.info(f"Scoring {len(to_score)} short_answer questions with GPT...")

    # Prepare items for batch scoring
    items = [
        {
            "question_id": r["question_id"],
            "question": r["question"],
            "reference": r.get("ground_truth", ""),
            "prediction": r.get("prediction", r.get("raw_output", ""))
        }
        for r in to_score
    ]

    # Get scores
    scores = judge.score_batch(items)

    # Build score map
    score_map = {
        it["question_id"]: s
        for it, s in zip(items, scores)
        if s is not None
    }

    failed = sum(1 for s in scores if s is None)
    if failed:
        logger.warning(f"{failed} items failed to score")

    # Apply scores
    updated = 0
    for r in records:
        if r["question_type"] != "short_answer":
            continue
        qid = r["question_id"]
        if qid not in score_map:
            continue

        gpt_s = score_map[qid]
        ev = r.get("evaluation_details") or {}
        ev["gpt4_score"] = gpt_s

        # Combine BERTScore and GPT score
        bert_s = ev.get("bert_score")
        if bert_s is not None:
            # Weighted combination: 60% BERT, 40% GPT
            final = 0.6 * bert_s + 0.4 * gpt_s
        else:
            final = gpt_s

        ev["final_score"] = final
        ev["score"] = final
        ev["correct"] = final >= 0.7

        r["evaluation_details"] = ev
        r["score"] = final
        r["correct"] = ev["correct"]
        updated += 1

    # Save updated results
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Recompute metrics
    from eatvid_benchmark.qa_evaluation.qa_metrics import QAMetrics, QAScoreConfig
    metrics = QAMetrics(QAScoreConfig())
    summary = metrics.compute_overall_metrics(records)

    # Save updated metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "task_name": "qa_evaluation",
            "num_samples": len(set(r["video_id"] for r in records)),
            "num_questions": len(records),
            "metrics": summary,
            "gpt_backfill": True
        }, f, indent=2, ensure_ascii=False)

    wacc = summary.get("weighted_accuracy", 0)
    logger.info(f"Backfill complete: {updated} questions updated. Weighted accuracy: {wacc:.4f}")

    return {
        "status": "success",
        "questions_updated": updated,
        "questions_failed": failed,
        "weighted_accuracy": wacc
    }


# ─────────────────────────── Convenience Functions ───────────────────────

def create_gpt_judge_from_config(score_config: Any) -> Optional[GPTJudge]:
    """
    Create GPTJudge instance from QAScoreConfig if enabled.

    This allows seamless integration with existing evaluation code.

    Args:
        score_config: QAScoreConfig instance with potential gpt4_judge settings

    Returns:
        GPTJudge instance if enabled, None otherwise
    """
    if not getattr(score_config, 'enable_gpt4_judge', False):
        return None

    # Extract GPT config from score_config
    config = GPTJudgeConfig(
        api_key=getattr(score_config, 'gpt4_api_key', None),
        base_url=getattr(score_config, 'gpt4_base_url', DEFAULT_API_BASE),
        model=getattr(score_config, 'gpt4_model', DEFAULT_MODEL),
        batch_size=getattr(score_config, 'gpt4_batch_size', 5)
    )

    return GPTJudge(config)


__all__ = [
    'GPTJudge',
    'GPTJudgeConfig',
    'batch_backfill_gpt_scores',
    'create_gpt_judge_from_config'
]
