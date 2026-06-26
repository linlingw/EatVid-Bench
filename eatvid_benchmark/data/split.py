"""
Data splitting utilities for eating behavior dataset.
Supports stratified splitting by subject/scene to avoid data leakage.
"""

import os
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SplitInfo:
    """Information about a data split."""
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    representative_ids: List[str]  # Subset for closed-source API evaluation


class DataSplitter:
    """
    Split video dataset into train/val/test sets.
    Ensures videos from the same subject don't appear in different splits.

    Supports two video discovery modes:
    1. From annotation_root/metadata/{video_id}/ directories
    2. From annotation task directories (e.g., annotation_root/bite/{video_id}/)
    """

    def __init__(
        self,
        annotation_root: str,
        video_root: Optional[str] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        representative_size: int = 100
    ):
        self.annotation_root = Path(annotation_root)
        self.video_root = Path(video_root) if video_root else None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.representative_size = representative_size

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"

    def discover_videos(self) -> List[str]:
        """
        Discover all video IDs from annotation directory.

        Discovery priority:
        1. If video_root is set, find videos there
        2. Check annotation_root/metadata/{video_id}/ directories
        3. Check annotation task directories (bite, action_timeline, etc.)
        """
        video_ids = set()

        # Priority 1: Check video_root if specified
        if self.video_root and self.video_root.exists():
            for item in self.video_root.iterdir():
                if item.is_file() and item.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Video file directly in video_root
                    video_ids.add(item.stem)
                elif item.is_dir():
                    # Video in subdirectory: video_root/{video_id}/{video_id}.mp4
                    for video_file in item.glob('*.mp4'):
                        if video_file.stem == item.name:
                            video_ids.add(item.name)

        # Priority 2: Check metadata directory
        metadata_dir = self.annotation_root / "metadata"
        if metadata_dir.exists():
            for video_dir in metadata_dir.iterdir():
                if video_dir.is_dir() and not video_dir.name.startswith('.'):
                    # Check if video file exists
                    video_file = video_dir / f"{video_dir.name}.mp4"
                    if video_file.exists():
                        video_ids.add(video_dir.name)
                    else:
                        # Check for any video file
                        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                            if list(video_dir.glob(f"*{ext}")):
                                video_ids.add(video_dir.name)
                                break

        # Priority 3: Check annotation task directories
        task_dirs = [
            "bite", "action_timeline", "facial_expression_sequence",
            "behavioral_health_analysis", "chewing", "emotion",
            "intake", "pace", "pose", "segments"
        ]
        for task_name in task_dirs:
            task_dir = self.annotation_root / task_name
            if task_dir.exists():
                for video_dir in task_dir.iterdir():
                    if video_dir.is_dir() and not video_dir.name.startswith('.'):
                        video_ids.add(video_dir.name)

        return sorted(list(video_ids))

    def extract_subject_id(self, video_id: str) -> str:
        """
        Extract subject identifier from video ID for stratified splitting.
        This ensures videos from the same subject don't leak across splits.

        Supported naming conventions:
        1. 'task{N}' -> Each video is independent (no subject grouping)
        2. 'subject{N}_session{M}' or 's{N}_sess{M}' -> Group by subject N
        3. '{subject}_{session}_{other}' -> Group by first part
        4. 'P{N}_{anything}' or 'participant{N}_{anything}' -> Group by participant N

        Args:
            video_id: The video identifier string

        Returns:
            Subject identifier for grouping
        """
        # Pattern 1: subject{N}_session{M} or s{N}_sess{M}
        match = re.match(r'^(subject|s)(\d+)[-_]', video_id, re.IGNORECASE)
        if match:
            return f"subject_{match.group(2)}"

        # Pattern 2: P{N}_ or participant{N}_
        match = re.match(r'^(p|participant)(\d+)[-_]', video_id, re.IGNORECASE)
        if match:
            return f"participant_{match.group(2)}"

        # Pattern 3: {name}_{session}_{...} - group by first part
        parts = re.split(r'[-_]', video_id)
        if len(parts) >= 2:
            first_part = parts[0].lower()
            # Check if first part looks like a subject identifier
            if re.match(r'^[a-zA-Z]+\d+$', first_part):
                return first_part

        # Pattern 4: task{N} format - treat each as independent
        # This is the default for the current dataset
        if re.match(r'^task\d+$', video_id, re.IGNORECASE):
            # For task-based naming, each video is independent
            # If you want to group tasks, modify this logic
            return video_id

        # Default: treat each video as independent subject
        return video_id

    def group_by_subject(self, video_ids: List[str]) -> Dict[str, List[str]]:
        """Group video IDs by subject."""
        groups = {}
        for vid in video_ids:
            subject = self.extract_subject_id(vid)
            if subject not in groups:
                groups[subject] = []
            groups[subject].append(vid)
        return groups
    
    def split(self, video_ids: Optional[List[str]] = None) -> SplitInfo:
        """
        Perform stratified split by subject.
        
        Returns:
            SplitInfo with train/val/test/representative video IDs
        """
        if video_ids is None:
            video_ids = self.discover_videos()
        
        random.seed(self.seed)
        
        # Group by subject for stratified splitting
        subject_groups = self.group_by_subject(video_ids)
        subjects = list(subject_groups.keys())
        random.shuffle(subjects)
        
        n_subjects = len(subjects)
        n_train = int(n_subjects * self.train_ratio)
        n_val = int(n_subjects * self.val_ratio)
        
        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train:n_train + n_val]
        test_subjects = subjects[n_train + n_val:]
        
        # Collect video IDs for each split
        train_ids = [vid for s in train_subjects for vid in subject_groups[s]]
        val_ids = [vid for s in val_subjects for vid in subject_groups[s]]
        test_ids = [vid for s in test_subjects for vid in subject_groups[s]]
        
        # Select representative subset from test set
        representative_ids = self._select_representative(test_ids)
        
        return SplitInfo(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            representative_ids=representative_ids
        )
    
    def _select_representative(self, test_ids: List[str]) -> List[str]:
        """Select challenging samples for closed-source API evaluation."""
        # For now, random sample; can be enhanced with difficulty scoring
        n = min(self.representative_size, len(test_ids))
        return random.sample(test_ids, n)
    
    def save_split(self, split_info: SplitInfo, output_path: str):
        """Save split information to JSON file."""
        data = {
            "train": split_info.train_ids,
            "val": split_info.val_ids,
            "test": split_info.test_ids,
            "representative": split_info.representative_ids,
            "config": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio,
                "seed": self.seed
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_split(split_path: str) -> SplitInfo:
        """Load split information from JSON file."""
        with open(split_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return SplitInfo(
            train_ids=data["train"],
            val_ids=data["val"],
            test_ids=data["test"],
            representative_ids=data.get("representative", [])
        )

