# EatVid-Bench

> A Multimodal Fine-Grained Eating Behavior Video Benchmark

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/linlingw/EatVid-Bench)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-blue)](LICENSE)

## 📖 Overview

EatVid-Bench is a comprehensive benchmark for evaluating video understanding models on fine-grained eating behavior analysis. It includes:

- **700+ eating-related videos** with structured annotations
- **4,000+ multimodal QA samples** covering 7 task categories and 3 difficulty levels
- **Multiple question types**: single-choice, multi-choice, true-false, fill-in-blank, short-answer
- **Comprehensive annotations**: bite detection, chewing analysis, food composition, pace estimation, facial expressions, action timelines, and behavioral health indicators

## 🗂️ Access

- **Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/linlingw/EatVid-Bench)
- **Benchmark Code**: This repository
- **Documentation**: See [DATASET.md](docs/DATASET.md) and [EVALUATION.md](docs/EVALUATION.md)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/linlingw/EatVid-Bench.git
cd EatVid-Bench
pip install -r requirements.txt
```

### Download Dataset

```bash
# Option 1: Using HuggingFace CLI
pip install huggingface_hub
huggingface-cli download linlingw/EatVid-Bench --local-dir ./data

# Option 2: Manual download
# Visit https://huggingface.co/datasets/linlingw/EatVid-Bench
```

### Run Evaluation

```bash
# Basic evaluation on test split
python -m eatvid_benchmark.run_qa_baseline \
  --model qwen2_5_vl_7b \
  --split test \
  --output_dir ./results

# With specific categories
python -m eatvid_benchmark.run_qa_baseline \
  --model qwen2_5_vl_7b \
  --categories bite,composition \
  --output_dir ./results

# With GPT Judge for short answers (optional)
export GPT_JUDGE_API_KEY="your-api-key"
python -m eatvid_benchmark.run_qa_baseline \
  --model qwen2_5_vl_7b \
  --enable_gpt_judge \
  --output_dir ./results
```

## 📊 Benchmark Structure

### Task Categories

| Category | Description | # Questions |
|----------|-------------|-------------|
| `bite` | Bite detection and counting | ~800 |
| `chewing` | Chewing frequency and duration | ~400 |
| `composition` | Food composition analysis | ~600 |
| `pace` | Eating pace and rhythm | ~400 |
| `facial_expression` | Facial expression changes | ~500 |
| `action_timeline` | Action event timeline | ~500 |
| `behavioral_health` | Behavioral health indicators | ~800 |
| `metadata` | Video metadata questions | ~200 |

### Question Types

- **Single Choice** (1.0×): Choose one correct option
- **Multi Choice** (2.0×): Choose all correct options
- **True/False** (0.5×): Yes/No questions
- **Fill Blank** (1.5×): Fill in the missing value
- **Short Answer** (5.0×): Open-ended text response

### Difficulty Levels

- **L1** (1.0×): Basic visual understanding
- **L2** (2.0×): Temporal reasoning required
- **L3** (3.0×): Complex multi-step reasoning

## 📈 Metrics

The benchmark uses weighted scoring to account for question difficulty and importance:

- **Accuracy**: Percentage of correctly answered questions
- **Weighted Accuracy**: Score weighted by difficulty, category, and question type
- **Type-wise Accuracy**: Breakdown by question type
- **Category-wise Accuracy**: Breakdown by task category

### Short Answer Evaluation

Short answer questions are evaluated using:
1. **BERTScore** (default): Semantic similarity between prediction and reference
2. **GPT Judge** (optional): LLM-based semantic evaluation
3. **Hybrid Mode**: `0.6 × BERTScore + 0.4 × GPT_Score`

## 🔧 Configuration

### Model Configuration

Edit `configs/benchmark_config.yaml` to configure model parameters:

```yaml
models:
  qwen2_5_vl_7b:
    video_use_nframes: True
    video_nframes: 64
    video_fps: 4.0
    video_max_pixels: 151200
```

### GPT Judge Configuration

```bash
export GPT_JUDGE_API_KEY="sk-..."
export GPT_JUDGE_MODEL="gpt-4o-mini"  # optional
export GPT_JUDGE_BASE_URL="https://api.openai.com/v1"  # optional
```

## 📁 Repository Structure

```
EatVid-Bench/
├── README.md                 # This file
├── LICENSE                   # CC BY-NC 4.0
├── requirements.txt          # Python dependencies
├── docs/
│   ├── DATASET.md           # Dataset documentation
│   ├── EVALUATION.md        # Evaluation protocol
│   └── CATEGORIES.md        # Task category details
│
├── eatvid_benchmark/         # Main package
│   ├── qa_evaluation/       # QA evaluation core
│   ├── models/              # Model interfaces
│   ├── data/                # Data processing
│   └── utils/               # Utilities
│
├── finetune/                # Fine-tuning tools (optional)
├── configs/                 # Configuration files
└── examples/                # Usage examples
```

## 🤝 Supported Models

### Open-Source Models
- Qwen2.5-VL-7B
- Qwen2-VL-7B
- InternVL2-8B
- VideoLLaMA3-7B
- HumanOmni-7B
- InternVideo2.5
- LLaVA-NeXT-Video-7B
- VideoLLaMA2-7B
- Molmo2-8B
- MiniCPM-o-4.5

### Closed-Source API Models
- GPT-4o
- Gemini Flash
- Qwen VL API

## 📝 Citation

If you use EatVid-Bench in your research, please cite:

```bibtex
@inproceedings{eatvidbench2025,
  title={EatVid-Bench: A Multimodal Fine-Grained Eating Behavior Video Benchmark},
  author={...},
  booktitle={ECCV},
  year={2025}
}
```

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). See [LICENSE](LICENSE) for details.

The dataset is available for research purposes only. Commercial use requires explicit permission.

## 🙏 Acknowledgments

- The HuggingFace team for the dataset hosting platform
- The open-source video understanding community

## 📧 Contact

For questions and feedback, please open an issue on GitHub.

---

**[Project Page](https://github.com/linlingw/EatVid-Bench) | [Dataset](https://huggingface.co/datasets/linlingw/EatVid-Bench) | [Documentation](docs/)**
