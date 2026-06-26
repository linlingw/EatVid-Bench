"""
EatVid-Bench: A Multimodal Fine-Grained Eating Behavior Video Benchmark

This package provides evaluation tools and utilities for the EatVid-Bench benchmark.
"""

__version__ = "1.0.0"
__author__ = "EatVid-Bench Team"

from .qa_evaluation import (
    QADataset,
    QATask,
    QAScoreConfig,
    QAMetrics
)

__all__ = [
    'QADataset',
    'QATask',
    'QAScoreConfig',
    'QAMetrics',
    '__version__',
]
