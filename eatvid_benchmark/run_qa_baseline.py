"""
问答式评估主运行脚本

Usage:
    # 运行所有问题的评估
    python run_qa_baseline.py --model qwen2_5_vl_7b --split test --output_dir ./results

    # 指定问题类别
    python run_qa_baseline.py --model qwen2_5_vl_7b --categories bite,composition --output_dir ./results

    # 指定题型
    python run_qa_baseline.py --model qwen2_5_vl_7b --types single_choice,fill_blank --output_dir ./results

    # 指定难度
    python run_qa_baseline.py --model qwen2_5_vl_7b --difficulty L1,L2 --output_dir ./results

    # 调试模式（限制样本数）
    python run_qa_baseline.py --model qwen2_5_vl_7b --max_samples 5 --output_dir ./results
"""

import os
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import yaml
import subprocess

# Suppress common noisy warnings before importing heavy libraries
# 1. Tokenizers parallelism warning (triggered when forked processes exist)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# 2. Enable expandable CUDA memory segments to reduce fragmentation
#    (helps prevent OOM from memory fragmentation in long-running inference loops)
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Suppress specific transformers warnings about TF/JAX
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
warnings.filterwarnings('ignore', message='.*computation placer already registered.*')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Add the experiments directory to Python path
EXPERIMENTS_DIR = Path(__file__).parent.absolute()
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from qa_evaluation import (
    QADataset,
    QATask,
    QAScoreConfig
)
from data.split import DataSplitter
from models.opensource.qwen_vl import Qwen25VLModel, Qwen2VLModel
from models.opensource.internvl import InternVL2_8BModel
from models.opensource.videollama3 import VideoLLaMA3_7BModel
from models.opensource.humanomni7b import HumanOmni7BModel
from models.opensource.internvideo25 import InternVideo25Model
from models.opensource.llava_next_video import LLaVANextVideoModel
from models.opensource.videollama2 import VideoLLaMA2Model
from models.opensource.molmo2 import Molmo2Model
from models.opensource.minicpm_o_4_5 import MiniCPMo45Model
from models.closedsource.api_models import (
    Qwen3VL32BAPIModel,
    Qwen3535BA3BAPIModel,
    Gemini3FlashModel,
    Gemini25FlashModel,
    Qwen35A3BAPIModel,
    SeedAPIModel
)
from models.baseline_models import RandomGuessBaseline, HumanSimulationBaseline


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default score configuration
DEFAULT_SCORE_CONFIG = QAScoreConfig(
    difficulty_weights={
        'L1': 1.0,
        'L2': 2.0,
        'L3': 3.0
    },
    category_weights={
        'action_timeline': 1.0,
        'bite': 1.0,
        'facial_expression': 1.0,
        'composition': 1.0,
        'pace': 1.0,
        'metadata': 0.5,
        'behavioral_health': 2.0  # 更高权重
    },
    question_type_base_scores={
        'single_choice': 1.0,
        'multi_choice': 2.0,
        'true_false': 0.5,
        'fill_blank': 1.5,
        'short_answer': 5.0
    },
    short_answer_method='bert_score',
    bert_score_threshold=0.7,
    enable_gpt4_judge=False
)


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_available_gpus():
    """Get list of available GPU devices."""
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        if result.returncode == 0:
            return [str(i) for i in range(len(result.stdout.strip().split('\n'))) if result.stdout.strip()]
        return []
    except Exception:
        return []

def setup_gpu_environment(devices):
    """Set up GPU environment variables."""
    if devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
        logger.info(f"Using GPUs: {devices}")
    else:
        # Use all available GPUs
        available_gpus = get_available_gpus()
        if available_gpus:
            devices = ','.join(available_gpus)
            os.environ['CUDA_VISIBLE_DEVICES'] = devices
            logger.info(f"Using all available GPUs: {devices}")

def get_model(model_name: str, config: dict, local_model_path: Optional[str] = None):
    """
    Initialize model by name.
    
    Args:
        model_name: Name of the model (e.g., 'qwen2_5_vl_7b')
        config: Experiment configuration
        local_model_path: Override local model path
    """
    # Base model configuration
    model_config = {
        'device': 'cuda',
        'dtype': 'bfloat16',
        'use_flash_attention': True,
        'max_frames': config.get('sampling', {}).get('long_term', {}).get('max_frames', 64)
    }
    
    # Get model-specific configuration from config file
    model_specific_config = config.get('models', {}).get(model_name, {})
    if model_specific_config:
        # Qwen2-VL style params
        if 'video_use_nframes' in model_specific_config:
            # Ensure boolean type (handle string "True"/"False" from YAML)
            val = model_specific_config['video_use_nframes']
            if isinstance(val, str):
                model_config['video_use_nframes'] = val.lower() in ('true', 'yes', '1', 'on')
            else:
                model_config['video_use_nframes'] = bool(val)
        if 'video_nframes' in model_specific_config:
            model_config['video_nframes'] = model_specific_config['video_nframes']
        if 'video_fps' in model_specific_config:
            model_config['video_fps'] = model_specific_config['video_fps']
        if 'video_max_pixels' in model_specific_config:
            model_config['video_max_pixels'] = model_specific_config['video_max_pixels']
        # InternVL style params
        if 'internvl_max_tiles' in model_specific_config:
            model_config['internvl_max_tiles'] = model_specific_config['internvl_max_tiles']
        # GPU allocation: specify which GPU to use (avoids multi-card OOM)
        if 'gpu_id' in model_specific_config:
            model_config['gpu_id'] = model_specific_config['gpu_id']
    
    # Get local path
    if local_model_path is None:
        local_paths = config.get('models', {}).get('local_paths', {})
        local_model_path = local_paths.get(model_name)
    
    if model_name == 'qwen2_5_vl_7b':
        return Qwen25VLModel(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'qwen2_vl_7b':
        return Qwen2VLModel(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'internvl2_8b':
        return InternVL2_8BModel(
            model_name="OpenGVLab/InternVL2-8B",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'videollama3_7b':
        return VideoLLaMA3_7BModel(
            model_name="DAMO-NLP-SG/VideoLLaMA3-7B",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'humanomni_7b':
        return HumanOmni7BModel(
            model_name="myroot/HumanOmni-7B-Video",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'internvideo2_5':
        return InternVideo25Model(
            model_name="OpenGVLab/InternVideo2_5_Chat_8B",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'llava_next_video':
        return LLaVANextVideoModel(
            model_name="swift/LLaVA-NeXT-Video-7B-hf",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'videollama2_7b':
        # Add VideoLLaMA2 specific parameters
        if model_specific_config:
            model_config['video_nframes'] = model_specific_config.get('video_nframes', 16)
            model_config['video_sampling_mode'] = model_specific_config.get('video_sampling_mode', 'uniform')
        return VideoLLaMA2Model(
            model_name="DAMO-NLP-SG/VideoLLaMA2-7B",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'molmo2_8b':
        # Add Molmo2-8B specific parameters
        if model_specific_config:
            model_config['fps'] = model_specific_config.get('fps', 2.0)
        return Molmo2Model(
            model_name="allenai/Molmo2-8B",
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'minicpm_o_4_5':
        return MiniCPMo45Model(
            model_name="OpenBMB/MiniCPM-o-4_5",
            config=model_config,
            local_path=local_model_path
        )
    # API-based models
    elif model_name == 'qwen3_vl_32b_api':
        return Qwen3VL32BAPIModel(
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'qwen3_5_35b_a3b_api':
        return Qwen3535BA3BAPIModel(
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'gemini3_flash':
        return Gemini3FlashModel(
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'gemini25_flash':
        return Gemini25FlashModel(
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'qwen35_a3b_api':
        return Qwen35A3BAPIModel(
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'seed_api':
        return SeedAPIModel(
            config=model_config,
            local_path=local_model_path
        )
    # Baseline models
    elif model_name == 'random_guess':
        return RandomGuessBaseline(
            config=model_config,
            local_path=local_model_path
        )
    elif model_name == 'human_simulation':
        return HumanSimulationBaseline(
            config=model_config,
            local_path=local_model_path
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: qwen2_5_vl_7b, qwen2_vl_7b, internvl2_8b, videollama3_7b, humanomni_7b, internvideo2_5, llava_next_video, videollama2_7b, molmo2_8b, minicpm_o_4_5, "
            f"qwen3_vl_32b_api, qwen3_5_35b_a3b_api, gemini3_flash, gemini25_flash, qwen35_a3b_api, seed_api, "
            f"random_guess, human_simulation"
        )


def parse_list_arg(arg: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated list argument."""
    if arg is None:
        return None
    return [item.strip() for item in arg.split(',')]


def run_qa_evaluation(
    model,
    model_name: str,
    split: str,
    config: dict,
    output_dir: str,
    split_info,
    categories: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    score_config: Optional[QAScoreConfig] = None,
    chunk_size: Optional[int] = None,
    batch_mode: bool = False,
    batch_by: str = "all",
    video_length_filter: Optional[float] = None,
    task_ids: Optional[List[str]] = None
):
    """Run QA evaluation."""
    logger.info(f"Running QA evaluation on {split} split")

    # Initialize dataset
    annotation_root = Path(config['data']['annotation_root'])
    video_root = Path(config['data']['video_root']) if config['data'].get('video_root') else None

    # split='all' 表示对整个数据集评估，不按 train/val/test 过滤
    effective_split_info = None if split == 'all' else split_info

    dataset = QADataset(
        annotation_root=str(annotation_root),
        video_root=str(video_root) if video_root else None,
        split=split,
        split_info=effective_split_info,
        categories=categories,
        question_types=question_types,
        difficulties=difficulties,
        video_length_filter=video_length_filter,
        task_ids=task_ids
    )
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    logger.info(f"Total questions: {stats['num_questions']}")
    logger.info(f"By type: {stats['by_type']}")
    logger.info(f"By category: {stats['by_category']}")
    logger.info(f"By difficulty: {stats['by_difficulty']}")
    
    # Prepare samples
    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples for debugging")
    
    if not samples:
        logger.warning("No valid samples found!")
        return None
    
    # Initialize QA task
    qa_task = QATask(
        model=model,
        config=config,
        score_config=score_config or DEFAULT_SCORE_CONFIG
    )
    
    # 设置批量模式（如果启用）
    if batch_mode:
        qa_task.set_batch_mode(enabled=True, batch_by=batch_by)
        logger.info(f"Batch mode enabled: grouping questions by {batch_by}")
    
    # Run evaluation
    if batch_mode and batch_by != 'single':
        logger.info(f"Using batch evaluation mode (grouping by {batch_by}, one call per video)")
        evaluation = qa_task.run_batch_mode(samples, show_progress=True)
    else:
        logger.info("Using single question evaluation mode (one call per question)")
        evaluation = qa_task.run_batch(samples, show_progress=True)
    
    # Save results
    task_output_dir = Path(output_dir) / 'qa_evaluation' / split
    qa_task.save_results(evaluation, str(task_output_dir), chunk_size=chunk_size)
    
    # Generate and print report
    report = qa_task.generate_report(evaluation)
    print(report)
    
    return evaluation


def main():
    parser = argparse.ArgumentParser(
        description='Run QA-based evaluation for eating behavior analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all questions with Qwen2.5-VL-7B
  python run_qa_baseline.py --model qwen2_5_vl_7b --output_dir ./results

  # Run with MiniCPM-o-4_5
  python run_qa_baseline.py --model minicpm_o_4_5 --output_dir ./results

  # Run specific categories
  python run_qa_baseline.py --model qwen2_5_vl_7b --categories bite,composition --output_dir ./results

  # Run specific question types
  python run_qa_baseline.py --model qwen2_5_vl_7b --types single_choice,fill_blank --output_dir ./results

  # Run with limited samples for debugging
  python run_qa_baseline.py --model qwen2_5_vl_7b --max_samples 5 --output_dir ./results
        """
    )
    parser.add_argument('--config', type=str, default='config/experiment_config.yaml',
                       help='Path to experiment config file')
    parser.add_argument('--model', type=str, default='qwen2_5_vl_7b',
                       choices=['qwen2_5_vl_7b', 'qwen2_vl_7b',
                                'internvl2_8b', 'videollama3_7b', 'humanomni_7b',
                                'internvideo2_5', 'llava_next_video', 'videollama2_7b',
                                'molmo2_8b', 'minicpm_o_4_5',
                                'qwen3_vl_32b_api', 'qwen3_5_35b_a3b_api', 'gemini3_flash',
                                'gemini25_flash', 'qwen35_a3b_api', 'seed_api',
                                'random_guess', 'human_simulation'],
                       help='Model to use for evaluation. '
                            'Available: qwen2_5_vl_7b, qwen2_vl_7b, '
                            'internvl2_8b, videollama3_7b, humanomni_7b, '
                            'internvideo2_5, llava_next_video, videollama2_7b, '
                            'molmo2_8b, minicpm_o_4_5, '
                            'qwen3_vl_32b_api, qwen3_5_35b_a3b_api, gemini3_flash, '
                            'gemini25_flash (MetaChat), qwen35_a3b_api (DashScope), seed_api (Volcengine), '
                            'random_guess (baseline), human_simulation (baseline)')
    parser.add_argument('--local_model_path', type=str, default=None,
                       help='Local path to model (overrides config file)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test', 'representative', 'all'],
                       help='Data split to evaluate on. Use "all" to evaluate on the entire dataset '
                            'without train/val/test splitting (recommended for benchmarking)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results (required)')
    parser.add_argument('--categories', type=str, default=None,
                       help='Comma-separated list of categories to evaluate '
                            '(action_timeline, bite, facial_expression, composition, pace, metadata, behavioral_health)')
    parser.add_argument('--types', type=str, default=None,
                       help='Comma-separated list of question types '
                            '(single_choice, multi_choice, true_false, fill_blank, short_answer)')
    parser.add_argument('--difficulty', type=str, default=None,
                       help='Comma-separated list of difficulty levels (L1, L2, L3)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for debugging)')
    parser.add_argument('--bert_threshold', type=float, default=0.7,
                       help='BERTScore threshold for short answer evaluation')
    parser.add_argument('--chunk_size', type=int, default=None,
                       help='If specified, save qa_evaluation_full.json in chunks of this many videos. '
                            'Useful for large datasets (e.g., --chunk_size 100 for 700 videos '
                            'creates 7 chunk files). qa_metrics.json and qa_results.jsonl are '
                            'always saved as single files.')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Enable batch mode: group all questions into one prompt per video '
                            '(more efficient, one model call per video instead of one per question). '
                            'Use --batch_by to specify grouping method.')
    parser.add_argument('--batch_by', type=str, default='single', choices=['single', 'all', 'difficulty', 'category'],
                       help='How to group questions: single (one call per question, default), all (one prompt), difficulty (by L1/L2/L3), category (by task type)')
    parser.add_argument('--video_length_filter', type=float, default=None,
                       help='Filter videos by length (in seconds). Positive value: skip videos shorter than this. Negative value: skip videos longer than absolute value.')
    parser.add_argument('--devices', type=str, default=None, help='Comma-separated list of GPU devices to use (e.g., 0,1,2)')
    parser.add_argument('--max_memory', type=str, default='45GB', help='Maximum memory per GPU (e.g., 45GB)')
    parser.add_argument('--task_id_path', type=str, default=None,
                       help='Path to a JSON file containing task IDs to evaluate '
                            '(e.g., {"worst_task_ids": ["task136", "task467", ...]}). '
                            'If specified, only tasks in the list will be evaluated. '
                            'video_length_filter still applies; tasks skipped by it are silently ignored.')
    
    args = parser.parse_args()

    # Set up GPU environment
    setup_gpu_environment(args.devices)

    # Load config
    config = load_config(args.config)

    # Parse list arguments
    categories = parse_list_arg(args.categories)
    question_types = parse_list_arg(args.types)
    difficulties = parse_list_arg(args.difficulty)

    # Load task_ids from JSON if specified (CLI arg takes priority over config)
    task_id_path = args.task_id_path or config.get('data', {}).get('task_id_path')
    task_ids: Optional[List[str]] = None
    if task_id_path:
        with open(task_id_path, 'r', encoding='utf-8') as f:
            task_id_data = json.load(f)
        task_ids = task_id_data.get('worst_task_ids', [])
        logger.info(f"Task ID filter loaded from {task_id_path}: {len(task_ids)} tasks "
                    f"(worst_task_count={task_id_data.get('worst_task_count', len(task_ids))})")
    
    # Log configuration
    logger.info(f"Model: {args.model}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max memory per GPU: {args.max_memory}")
    if categories:
        logger.info(f"Categories: {categories}")
    if question_types:
        logger.info(f"Question types: {question_types}")
    if difficulties:
        logger.info(f"Difficulties: {difficulties}")
    if args.video_length_filter is not None:
        if args.video_length_filter > 0:
            logger.info(f"Video length filter: skipping videos shorter than {args.video_length_filter} seconds")
        else:
            logger.info(f"Video length filter: skipping videos longer than {abs(args.video_length_filter)} seconds")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # Create or load data split
    annotation_root = Path(config['data']['annotation_root'])
    split_path = output_dir / 'data_split.json'
    
    splitter = DataSplitter(
        annotation_root=str(annotation_root),
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        seed=config['data']['split_seed']
    )
    split_info = splitter.split()
    splitter.save_split(split_info, str(split_path))
    
    if args.split == 'all':
        total_videos = len(split_info.train_ids) + len(split_info.val_ids) + len(split_info.test_ids)
        logger.info(f"Running on ALL videos (no split filtering). "
                   f"Total: {total_videos} (train={len(split_info.train_ids)}, "
                   f"val={len(split_info.val_ids)}, test={len(split_info.test_ids)})")
    else:
        logger.info(f"Data split: train={len(split_info.train_ids)}, "
                   f"val={len(split_info.val_ids)}, test={len(split_info.test_ids)}")
    
    # Initialize model
    logger.info(f"Initializing model: {args.model}")
    model = get_model(args.model, config, args.local_model_path)
    # Add max_memory to model config if specified
    if hasattr(model, 'config'):
        model.config['max_memory'] = args.max_memory
    model.load()
    
    # Configure score config
    score_config = QAScoreConfig(
        difficulty_weights=DEFAULT_SCORE_CONFIG.difficulty_weights,
        category_weights=DEFAULT_SCORE_CONFIG.category_weights,
        question_type_base_scores=DEFAULT_SCORE_CONFIG.question_type_base_scores,
        short_answer_method='bert_score',
        bert_score_threshold=args.bert_threshold,
        enable_gpt4_judge=False
    )
    
    # Run evaluation
    try:
        evaluation = run_qa_evaluation(
            model=model,
            model_name=args.model,
            split=args.split,
            config=config,
            output_dir=str(output_dir),
            split_info=split_info,
            categories=categories,
            question_types=question_types,
            difficulties=difficulties,
            max_samples=args.max_samples,
            score_config=score_config,
            chunk_size=args.chunk_size,
            batch_mode=args.batch_mode,
            batch_by=args.batch_by,
            video_length_filter=args.video_length_filter,
            task_ids=task_ids
        )
        
        if evaluation:
            # Save summary
            summary = {
                'model': args.model,
                'split': args.split,
                'categories': categories,
                'question_types': question_types,
                'difficulties': difficulties,
                'metrics': evaluation.metrics
            }
            summary_path = output_dir / 'summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\nEvaluation complete. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Unload model
        model.unload()


if __name__ == '__main__':
    main()