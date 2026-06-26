# EatVid-Bench Evaluation Protocol

## Overview

This document describes the evaluation protocol for EatVid-Bench, including metric calculation, scoring methods, and result submission format.

## Evaluation Setup

### Prerequisites

1. Download the dataset from HuggingFace
2. Install dependencies: `pip install -r requirements.txt`
3. Configure model settings in `configs/benchmark_config.yaml`

### Running Evaluation

#### Basic Evaluation

```bash
python -m eatvid_benchmark.run_qa_baseline \
  --model qwen2_5_vl_7b \
  --split test \
  --output_dir ./results
```

#### Evaluation with Options

```bash
# Specific categories only
python -m eatvid_benchmark.run_qa_baseline \
  --model qwen2_5_vl_7b \
  --categories bite,composition \
  --output_dir ./results

# Specific difficulty levels
python -m eatvid_benchmark.run_qa_baseline \
  --model qwen2_5_vl_7b \
  --difficulty L1,L2 \
  --output_dir ./results

# With GPT Judge for short answers
export GPT_JUDGE_API_KEY="your-api-key"
python -m eatvid_benchmark.run_qa_baseline \
  --model qwen2_5_vl_7b \
  --enable_gpt_judge \
  --output_dir ./results
```

## Metrics

### Weighted Accuracy

The primary metric is weighted accuracy, which accounts for:

1. **Question Type Weight**: Different question types have different base scores
2. **Difficulty Weight**: L3 questions are worth 3× L1 questions
3. **Category Weight**: Behavioral health questions have 2× weight

#### Score Calculation

For each question:

```
question_score = base_score × difficulty_weight × category_weight
weighted_score = question_score × model_score (0-1)
```

#### Final Metrics

- **Accuracy**: `correct_questions / total_questions`
- **Weighted Accuracy**: `sum(weighted_scores) / sum(max_possible_scores)`
- **Type-wise Accuracy**: Accuracy per question type
- **Category-wise Accuracy**: Accuracy per task category

### Short Answer Evaluation

Short answer questions support three evaluation modes:

#### 1. BERTScore (Default)

Uses semantic similarity between prediction and reference:

```python
score = bert_score(prediction, reference)
final_score = 1.0 if score >= 0.7 else score / 0.7
```

#### 2. GPT Judge (Optional)

Uses LLM to evaluate semantic correctness:

```python
score = gpt_judge.score_single(question, reference, prediction)
final_score = score  # 0.0 to 1.0
```

#### 3. Hybrid Mode

Combines both methods:

```python
final_score = 0.6 × bert_score + 0.4 × gpt_score
```

## Output Format

### Result Files

Evaluation produces the following files:

```
results/
├── qa_evaluation/
│   └── {split}/
│       ├── qa_metrics.json           # Summary metrics
│       ├── qa_results.jsonl          # Per-question results
│       └── qa_evaluation_full.json   # Complete evaluation
```

### qa_metrics.json

```json
{
  "task_name": "qa_evaluation",
  "num_samples": 100,
  "num_questions": 1500,
  "metrics": {
    "total_questions": 1500,
    "correct_questions": 1050,
    "accuracy": 0.70,
    "weighted_score": 2100.0,
    "max_score": 3000.0,
    "weighted_accuracy": 0.70,
    "by_type": {
      "single_choice": {
        "total": 600,
        "correct": 480,
        "accuracy": 0.80,
        "weighted_accuracy": 0.80
      }
    },
    "by_category": {
      "bite": {
        "total": 200,
        "correct": 160,
        "accuracy": 0.80,
        "weighted_accuracy": 0.80
      }
    },
    "by_difficulty": {
      "L1": {
        "total": 750,
        "correct": 600,
        "accuracy": 0.80,
        "weighted_accuracy": 0.80
      }
    }
  }
}
```

### qa_results.jsonl

Each line is a JSON object representing one question:

```json
{
  "question_id": "task001_q1",
  "video_id": "task001",
  "question_type": "single_choice",
  "difficulty": "L1",
  "category": "bite",
  "question": "How many times does the person take a bite?",
  "prediction": "B",
  "ground_truth": "B",
  "correct": true,
  "score": 1.0,
  "evaluation_details": {
    "prediction_clean": "B",
    "ground_truth_clean": "B"
  }
}
```

## Benchmark Leaderboard

### Baseline Results

| Model | Accuracy | Weighted Acc | Single | Multi | T/F | Fill | Short |
|-------|----------|--------------|--------|-------|-----|------|-------|
| Qwen2.5-VL-7B | 65.2% | 58.3% | 78.5% | 52.3% | 85.2% | 61.8% | 48.5% |
| InternVL2-8B | 62.8% | 55.7% | 75.2% | 48.9% | 82.1% | 58.3% | 45.2% |
| GPT-4o | 71.5% | 64.8% | 85.3% | 62.1% | 89.5% | 68.7% | 52.3% |

*Note: Results are preliminary and may vary with evaluation settings.*

### Submission Format

To submit results to the leaderboard:

1. Run evaluation on the test split
2. Ensure GPT Judge is enabled for short answers
3. Submit `qa_metrics.json` via the submission form

## Advanced Configuration

### Custom Weights

Modify score weights in `configs/benchmark_config.yaml`:

```yaml
scoring:
  difficulty_weights:
    L1: 1.0
    L2: 2.0
    L3: 3.0
  category_weights:
    bite: 1.0
    behavioral_health: 2.0
```

### GPT Judge Configuration

```bash
export GPT_JUDGE_API_KEY="sk-..."
export GPT_JUDGE_MODEL="gpt-4o-mini"
export GPT_JUDGE_BASE_URL="https://api.openai.com/v1"
export GPT_JUDGE_BATCH_SIZE=5
```

## Troubleshooting

### Common Issues

**Q: BERTScore fails to load**
A: Set `TRANSFORMERS_OFFLINE=1` to use cached models only, or install with `pip install bert-score`

**Q: GPT Judge returns errors**
A: Check your API key and ensure you have sufficient credits

**Q: Model runs out of memory**
A: Reduce `video_nframes` or use a smaller batch size

## Citation

If you use the benchmark evaluation code, please cite:

```bibtex
@inproceedings{eatvidbench2025,
  title={EatVid-Bench: A Multimodal Fine-Grained Eating Behavior Video Benchmark},
  booktitle={ECCV},
  year={2025}
}
```

---

**[Back to README](../README.md) | [Dataset Documentation](DATASET.md)**
