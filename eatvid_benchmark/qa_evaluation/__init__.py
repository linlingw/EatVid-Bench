"""
问答式评估模块 (QA-based Evaluation Module)

该模块提供了基于问答形式的视频理解评估框架，
支持选择题、判断题、填空题、简答题等多种题型。

主要组件：
- QADataset: 问答数据集加载
- QATask: 问答评估任务
- QAMetrics: 评估指标计算
- QAPrompts: Prompt模板
"""

from .qa_dataset import QAQuestion, QASample, QADataset
from .qa_metrics import QAMetrics, QAScoreConfig
from .qa_prompts import QAPromptTemplates
from .qa_task import QATask, QAResult, QAEvaluation

__all__ = [
    'QAQuestion',
    'QASample', 
    'QADataset',
    'QAMetrics',
    'QAScoreConfig',
    'QAPromptTemplates',
    'QATask',
    'QAResult',
    'QAEvaluation',
]
