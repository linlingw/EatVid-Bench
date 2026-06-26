# EatVid-Bench Dataset Documentation

## Overview

EatVid-Bench is a multimodal fine-grained eating behavior video benchmark designed for comprehensive evaluation of video understanding models. The dataset contains 700+ eating-related videos with rich annotations covering various aspects of eating behavior.

## Dataset Contents

### Video Data

- **Total Videos**: 700+
- **Average Duration**: 30-120 seconds
- **Video Quality**: HD (720p or higher)
- **Perspectives**: First-person (egocentric) and third-person views
- **Privacy**: All faces have been anonymized using state-of-the-art face detection and blurring

### Annotation Types

| Annotation | Description | File Format |
|-----------|-------------|-------------|
| **Metadata** | Video ID, duration, resolution, frame rate | JSON |
| **Bite Events** | Timestamps of food intake actions | JSON |
| **Chewing** | Chewing frequency and duration | JSON |
| **Food Composition** | Food types and proportions | JSON |
| **Eating Pace** | Eating speed and rhythm analysis | JSON |
| **Facial Expressions** | Expression change timeline | JSON |
| **Action Timeline** | Fine-grained action events | JSON |
| **Behavioral Health** | Health indicator analysis | JSON |

### QA Samples

The dataset includes 4,000+ multimodal QA samples covering:

#### Question Categories

- **Bite** (800 questions): Detection and counting of bite events
- **Chewing** (400 questions): Chewing frequency and duration analysis
- **Composition** (600 questions): Food composition and dietary balance
- **Pace** (400 questions): Eating speed and rhythm
- **Facial Expression** (500 questions): Emotional states during eating
- **Action Timeline** (500 questions): Temporal understanding of actions
- **Behavioral Health** (800 questions): Overall behavioral health indicators
- **Metadata** (200 questions): Basic video information

#### Question Types

| Type | Description | Weight |
|------|-------------|--------|
| Single Choice | Choose one correct option (A/B/C/D) | 1.0× |
| Multi Choice | Choose all correct options | 2.0× |
| True/False | Yes/No questions | 0.5× |
| Fill Blank | Fill in numerical/factual values | 1.5× |
| Short Answer | Open-ended text response | 5.0× |

#### Difficulty Levels

- **L1** (Basic): Direct visual understanding
- **L2** (Intermediate): Temporal reasoning required
- **L3** (Advanced): Multi-step reasoning and synthesis

## Data Access

### Download via HuggingFace

```bash
# Using HuggingFace CLI
pip install huggingface_hub
huggingface-cli download linlingw/EatVid-Bench --local-dir ./data

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="linlingw/EatVid-Bench",
    local_dir="./data",
    local_dir_use_symlinks=False
)
```

### Manual Download

Visit [HuggingFace Dataset](https://huggingface.co/datasets/linlingw/EatVid-Bench) to download files manually.

## Data Format

### Directory Structure

```
data/
├── annotations/
│   ├── metadata/
│   │   ├── {video_id}/{video_id}_metadata.json
│   ├── bite/
│   │   ├── {video_id}/{video_id}_bite.json
│   ├── chewing/
│   ├── composition/
│   ├── pace/
│   ├── facial_expression/
│   ├── action_timeline/
│   └── behavioral_health/
│
├── questions/
│   ├── {video_id}/{video_id}_questions.json
│
├── splits/
│   ├── train.json
│   ├── val.json
│   └── test.json
│
└── videos/
    ├── {video_id}.mp4
    └── ...
```

### Annotation Schema

Each annotation file follows a JSON schema. Example for bite annotation:

```json
{
  "video_id": "task001",
  "num_bites": 12,
  "duration": 45.2,
  "bite_events": [
    {
      "frame_index": 120,
      "timestamp": 4.0,
      "peak_frame": 125,
      "end_frame": 135,
      "hand": "Right",
      "confidence": 0.95
    }
  ]
}
```

### Question Schema

```json
{
  "questions": [
    {
      "question_id": "task001_q1",
      "video_id": "task001",
      "question": "How many times does the person take a bite?",
      "question_type": "single_choice",
      "category": "bite",
      "difficulty": "L1",
      "options": ["A. 10 times", "B. 12 times", "C. 15 times", "D. 18 times"],
      "answer": "B",
      "reference_answer": "The person takes a bite 12 times."
    }
  ]
}
```

## Data Splits

The benchmark provides three data splits:

- **Train** (70%): Model development and fine-tuning
- **Val** (15%): Hyperparameter tuning
- **Test** (15%): Final evaluation

Split files are available in `data/splits/` and contain lists of video IDs.

## License and Usage

- **License**: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial)
- **Usage**: Research and educational use only
- **Commercial Use**: Requires explicit permission

### Citation

If you use the dataset, please cite:

```bibtex
@inproceedings{eatvidbench2025,
  title={EatVid-Bench: A Multimodal Fine-Grained Eating Behavior Video Benchmark},
  booktitle={ECCV},
  year={2025}
}
```

## Privacy and Ethics

- All videos were collected with informed consent
- Faces have been anonymized using state-of-the-art detection and blurring
- The dataset does not contain sensitive personal information
- Access is restricted to research purposes

## Contact

For questions about the dataset, please open an issue on GitHub.

---

**[Back to README](../README.md) | [Evaluation Documentation](EVALUATION.md)**
