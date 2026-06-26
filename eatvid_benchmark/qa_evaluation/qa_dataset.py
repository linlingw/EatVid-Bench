"""
问答数据集加载模块

负责从标注文件中加载问答数据，支持数据划分和视频路径解析。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class QAQuestion:
    """单个问题"""
    question_id: str
    question_type: str  # single_choice, multi_choice, true_false, fill_blank, short_answer
    difficulty: str  # L1, L2, L3
    category: str  # action_timeline, bite, facial_expression, composition, pace, metadata, behavioral_health
    question: str
    options: Optional[List[str]] = None
    answer: str = ""
    reference_answer: Optional[str] = None
    answer_tolerance: Optional[Dict[str, Any]] = None
    answer_source: Optional[Dict[str, Any]] = None
    key_points: Optional[List[str]] = None
    evaluation: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAQuestion':
        """从字典创建问题对象"""
        return cls(
            question_id=data['question_id'],
            question_type=data['question_type'],
            difficulty=data['difficulty'],
            category=data['category'],
            question=data['question'],
            options=data.get('options'),
            answer=data.get('answer', ''),
            reference_answer=data.get('reference_answer'),
            answer_tolerance=data.get('answer_tolerance'),
            answer_source=data.get('answer_source'),
            key_points=data.get('key_points'),
            evaluation=data.get('evaluation')
        )


@dataclass
class QASample:
    """单个视频的所有问题"""
    video_id: str
    video_path: Optional[str]
    questions: List[QAQuestion]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_questions_by_type(self, question_type: str) -> List[QAQuestion]:
        """按题型筛选问题"""
        return [q for q in self.questions if q.question_type == question_type]
    
    def get_questions_by_category(self, category: str) -> List[QAQuestion]:
        """按类别筛选问题"""
        return [q for q in self.questions if q.category == category]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[QAQuestion]:
        """按难度筛选问题"""
        return [q for q in self.questions if q.difficulty == difficulty]


class QADataset:
    """
    问答数据集
    
    负责从标注目录加载问答数据，支持数据划分。
    """
    
    # 支持的问题类型
    QUESTION_TYPES = ['single_choice', 'multi_choice', 'true_false', 'fill_blank', 'short_answer']
    
    # 支持的问题类别
    CATEGORIES = ['action_timeline', 'bite', 'facial_expression', 'composition', 
                  'pace', 'metadata', 'behavioral_health']
    
    # 难度等级
    DIFFICULTIES = ['L1', 'L2', 'L3']
    
    def __init__(
        self,
        annotation_root: str,
        video_root: Optional[str] = None,
        split: str = 'test',
        split_info: Optional[Any] = None,
        categories: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        video_length_filter: Optional[float] = None,
        task_ids: Optional[List[str]] = None
    ):
        """
        初始化问答数据集

        Args:
            annotation_root: 标注文件根目录
            video_root: 视频文件根目录（可选，默认从metadata目录查找）
            split: 数据划分 (train/val/test)
            split_info: 数据划分信息
            categories: 要加载的类别（可选，默认加载全部）
            question_types: 要加载的题型（可选，默认加载全部）
            difficulties: 要加载的难度（可选，默认加载全部）
            video_length_filter: 视频长度过滤（秒）。正数：跳过短于该值的视频。负数：跳过长于绝对值的视频。
            task_ids: 限定只评估的 task_id 列表（可选）。若指定，仅处理列表中的 task。
                      video_length_filter 仍会生效；被长度过滤跳过的 task 不会报错，直接忽略。
        """
        self.annotation_root = Path(annotation_root)
        self.video_root = Path(video_root) if video_root else None
        self.split = split
        self.split_info = split_info

        # 筛选条件
        self.filter_categories = categories
        self.filter_question_types = question_types
        self.filter_difficulties = difficulties
        self.video_length_filter = video_length_filter
        self.filter_task_ids: Optional[set] = set(task_ids) if task_ids is not None else None

        # 加载数据
        self._samples: Dict[str, QASample] = {}
        self._load_data()
    
    def _get_video_length(self, video_path: str) -> Optional[float]:
        """获取视频长度（秒）"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps == 0:
                return None
            length = total_frames / fps
            cap.release()
            return length
        except Exception as e:
            logger.warning(f"Error getting video length for {video_path}: {e}")
            return None
    
    def _should_keep_video(self, video_path: Optional[str]) -> bool:
        """检查视频是否应该被保留（基于长度过滤）"""
        if self.video_length_filter is None or video_path is None:
            return True
        
        video_length = self._get_video_length(video_path)
        if video_length is None:
            return True
        
        if self.video_length_filter > 0:
            # 正数：跳过短于该值的视频
            if video_length < self.video_length_filter:
                logger.info(f"Skipping video {video_path} (length: {video_length:.2f}s < {self.video_length_filter}s)")
                return False
        else:
            # 负数：跳过长于绝对值的视频
            max_length = abs(self.video_length_filter)
            if video_length > max_length:
                logger.info(f"Skipping video {video_path} (length: {video_length:.2f}s > {max_length}s)")
                return False
        
        return True
    
    def _load_data(self):
        """加载问答数据"""
        questions_dir = self.annotation_root / 'questions'
        
        if not questions_dir.exists():
            logger.warning(f"Questions directory not found: {questions_dir}")
            return
        
        # 遍历所有任务目录
        for task_dir in questions_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            video_id = task_dir.name
            questions_file = task_dir / f"{video_id}_questions.json"
            
            if not questions_file.exists():
                logger.warning(f"Questions file not found: {questions_file}")
                continue
            
            # 检查是否在当前划分中
            if self.split_info and not self._is_in_split(video_id):
                continue

            # 检查是否在指定的 task_id 列表中
            if self.filter_task_ids is not None and video_id not in self.filter_task_ids:
                continue

            # 加载问题文件
            try:
                with open(questions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 解析问题
                questions = []
                for q_data in data.get('questions', []):
                    question = QAQuestion.from_dict(q_data)
                    
                    # 应用筛选条件
                    if not self._passes_filters(question):
                        continue
                    
                    questions.append(question)
                
                if not questions:
                    continue
                
                # 获取视频路径
                video_path = self._get_video_path(video_id)
                
                # 检查视频长度
                if not self._should_keep_video(video_path):
                    continue
                
                # 创建样本
                sample = QASample(
                    video_id=video_id,
                    video_path=video_path,
                    questions=questions,
                    metadata=data.get('metadata', {})
                )
                
                self._samples[video_id] = sample
                
            except Exception as e:
                logger.error(f"Error loading questions for {video_id}: {e}")
        
        logger.info(f"Loaded {len(self._samples)} samples for {self.split} split")
    
    def _is_in_split(self, video_id: str) -> bool:
        """检查视频是否在当前划分中"""
        if self.split_info is None:
            return True
        
        if self.split == 'train':
            return video_id in self.split_info.train_ids
        elif self.split == 'val':
            return video_id in self.split_info.val_ids
        elif self.split == 'test':
            return video_id in self.split_info.test_ids
        elif self.split == 'representative':
            return video_id in getattr(self.split_info, 'representative_ids', [])
        
        return True
    
    def _passes_filters(self, question: QAQuestion) -> bool:
        """检查问题是否通过筛选条件"""
        if self.filter_categories and question.category not in self.filter_categories:
            return False
        if self.filter_question_types and question.question_type not in self.filter_question_types:
            return False
        if self.filter_difficulties and question.difficulty not in self.filter_difficulties:
            return False
        return True
    
    def _get_video_path(self, video_id: str) -> Optional[str]:
        """获取视频文件路径"""
        # 优先使用video_root
        if self.video_root:
            video_path = self.video_root / f"{video_id}.mp4"
            if video_path.exists():
                return str(video_path)
        
        # 从metadata目录查找
        metadata_dir = self.annotation_root / 'metadata' / video_id
        if metadata_dir.exists():
            video_path = metadata_dir / f"{video_id}.mp4"
            if video_path.exists():
                return str(video_path)
        
        logger.warning(f"Video not found for {video_id}")
        return None
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __iter__(self) -> Iterator[QASample]:
        return iter(self._samples.values())
    
    def __getitem__(self, video_id: str) -> Optional[QASample]:
        return self._samples.get(video_id)
    
    def get_video_ids(self) -> List[str]:
        """获取所有视频ID"""
        return list(self._samples.keys())
    
    def get_all_questions(self) -> List[QAQuestion]:
        """获取所有问题"""
        questions = []
        for sample in self._samples.values():
            questions.extend(sample.questions)
        return questions
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        all_questions = self.get_all_questions()
        
        stats = {
            'num_videos': len(self._samples),
            'num_questions': len(all_questions),
            'by_type': {},
            'by_category': {},
            'by_difficulty': {}
        }
        
        for q in all_questions:
            # 按题型统计
            stats['by_type'][q.question_type] = stats['by_type'].get(q.question_type, 0) + 1
            # 按类别统计
            stats['by_category'][q.category] = stats['by_category'].get(q.category, 0) + 1
            # 按难度统计
            stats['by_difficulty'][q.difficulty] = stats['by_difficulty'].get(q.difficulty, 0) + 1
        
        return stats