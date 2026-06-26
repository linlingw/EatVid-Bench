# Fine-tuning Tools

This directory contains utilities for preparing training data and fine-tuning models on EatVid-Bench.

## Files

### generate_cot.py
Generates Chain-of-Thought (CoT) reasoning chains from annotations for each task.

**Usage:**
```bash
python generate_cot.py \
  --annotation_root /path/to/annotations \
  --task_id task001 \
  --print
```

### prepare_train_data.py
Prepares LLaMA-Factory format training data with CoT reasoning.

**Usage:**
```bash
python prepare_train_data.py \
  --task_ids_file train_task_ids.json \
  --annotation_root /path/to/annotations \
  --video_root /path/to/videos \
  --output_file train_data.json
```

### evaluate.py
Evaluation script for fine-tuned models.

**Requirements:**
- LLaMA-Factory installed
- Fine-tuned model checkpoint

**Usage:**
```bash
python evaluate.py \
  --model_path /path/to/checkpoint \
  --test_data /path/to/test.json
```

## Training Data Format

The generated training data follows LLaMA-Factory's video ShareGPT format:

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nQuestion text with options?"
      },
      {
        "from": "gpt",
        "value": "<thinking>\nChain of thought reasoning\n</thinking>\nAnswer"
      }
    ],
    "videos": ["/path/to/video.mp4"]
  }
]
```

## Notes

- These tools are optional for benchmark evaluation
- They are provided for researchers who want to fine-tune models on this dataset
- See the main README for benchmark evaluation instructions
