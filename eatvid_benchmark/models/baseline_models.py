"""
Baseline models for evaluation comparison.

Provides:
1. RandomGuessModel - Random chance baseline (lower bound)
2. HumanSimulationModel - Simulated human performance (upper bound)
"""

import random
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np

from models.base_model import BaseVideoModel, ModelOutput, ModelRegistry


class RandomGuessModel(BaseVideoModel):
    """
    Random guess baseline model.
    Provides a lower bound for comparison by randomly selecting answers.
    This represents the "difficulty reference" - how hard the task is.
    """

    def __init__(
        self,
        model_name: str = "random_guess",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None,
        seed: int = 42
    ):
        super().__init__(model_name, config)
        self.seed = seed
        self.rng = random.Random(seed)

    def load(self) -> None:
        """No loading needed for random guess"""
        self.is_loaded = True
        print(f"Random Guess model initialized with seed={self.seed}")

    def unload(self) -> None:
        """Nothing to unload"""
        self.is_loaded = False

    def _guess_single_choice(self, options: List[str]) -> str:
        """Randomly select one option"""
        if not options:
            return "A"
        return self.rng.choice(options)

    def _guess_multi_choice(self, options: List[str]) -> str:
        """Randomly select a subset of options"""
        if not options:
            return "A"
        # Randomly select 1 to len(options) options
        num_selected = self.rng.randint(1, len(options))
        selected = self.rng.sample(options, num_selected)
        return ",".join(sorted(selected))

    def _guess_true_false(self) -> str:
        """Randomly select True or False"""
        return self.rng.choice(["True", "False", "true", "false", "是", "否", "正确", "错误"])

    def _guess_fill_blank(self, ground_truth: str = None) -> str:
        """Generate a random number guess"""
        if ground_truth:
            try:
                gt_val = float(ground_truth)
                # Generate random number within reasonable range
                if gt_val == 0:
                    return str(self.rng.randint(-5, 10))
                else:
                    # Random within ±50% of ground truth or ±5
                    delta = max(abs(gt_val) * 0.5, 5)
                    return str(self.rng.uniform(gt_val - delta, gt_val + delta))
            except:
                pass
        return str(self.rng.randint(0, 20))

    def _guess_short_answer(self) -> str:
        """Generate a random short answer"""
        templates = [
            "视频中显示{}",
            "我观察到{}",
            "根据视频内容，{}",
            "{}",
            "可能是{}",
            "不确定，但{}"
        ]
        answers = [
            "有一些动作",
            "人物在吃东西",
            "没有明显变化",
            "有一些表情变化",
            "视频内容不清楚",
            "有几个片段",
            "动作比较明显",
            "不太确定"
        ]
        template = self.rng.choice(templates)
        answer = self.rng.choice(answers)
        return template.format(answer)

    def inference(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        fps: float = 2.0,
        video_path: Optional[str] = None,
        question_type: str = "single_choice",
        options: List[str] = None,
        ground_truth: str = None,
        **kwargs
    ) -> ModelOutput:
        """
        Generate a random guess based on question type.

        The question_type, options, and ground_truth can be passed via kwargs
        by the QATask evaluator.
        """
        if not self.is_loaded:
            self.load()

        # Extract question info from metadata if available
        metadata = kwargs.get('metadata', {})

        # Determine question type
        q_type = metadata.get('question_type', question_type)
        q_options = metadata.get('options', options)
        q_answer = metadata.get('ground_truth', ground_truth)

        # Generate guess based on type
        if q_type == 'single_choice':
            prediction = self._guess_single_choice(q_options or ['A', 'B', 'C', 'D'])
        elif q_type == 'multi_choice':
            prediction = self._guess_multi_choice(q_options or ['A', 'B', 'C', 'D'])
        elif q_type == 'true_false':
            prediction = self._guess_true_false()
        elif q_type == 'fill_blank':
            prediction = self._guess_fill_blank(q_answer)
        elif q_type == 'short_answer':
            prediction = self._guess_short_answer()
        else:
            # Default: random choice
            prediction = self._guess_single_choice(q_options or ['A', 'B', 'C', 'D'])

        return ModelOutput(
            raw_text=prediction,
            parsed_data=None,
            metadata={
                "model": self.model_name,
                "guess_type": q_type,
                "seed": self.seed
            }
        )

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def max_frames(self) -> int:
        return 0


class HumanSimulationModel(BaseVideoModel):
    """
    Simulated human performance baseline model.
    Provides an upper bound for comparison.

    Simulates a careful human annotator who can:
    - Watch the video multiple times
    - Pause and count frames
    - Use domain knowledge

    Performance targets (based on typical human annotation quality):
    - Single choice: 92% accuracy
    - Multi choice: 88% accuracy
    - True/False: 96% accuracy
    - Fill blank: 85% accuracy (with small numerical error)
    - Short answer: 75% BERTScore (semantically similar but not identical)
    """

    def __init__(
        self,
        model_name: str = "human_simulation",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None,
        seed: int = 42,
        # Accuracy settings
        single_choice_acc: float = 0.92,
        multi_choice_acc: float = 0.88,
        true_false_acc: float = 0.96,
        fill_blank_acc: float = 0.85,
        short_answer_quality: float = 0.75
    ):
        super().__init__(model_name, config)
        self.seed = seed
        self.rng = random.Random(seed)
        self.accuracy_settings = {
            'single_choice': single_choice_acc,
            'multi_choice': multi_choice_acc,
            'true_false': true_false_acc,
            'fill_blank': fill_blank_acc,
            'short_answer': short_answer_quality
        }

    def load(self) -> None:
        """No loading needed for simulation"""
        self.is_loaded = True
        print(f"Human Simulation model initialized with seed={self.seed}")
        print(f"  Target accuracies: SC={self.accuracy_settings['single_choice']:.0%}, "
              f"MC={self.accuracy_settings['multi_choice']:.0%}, "
              f"TF={self.accuracy_settings['true_false']:.0%}, "
              f"FB={self.accuracy_settings['fill_blank']:.0%}")

    def unload(self) -> None:
        """Nothing to unload"""
        self.is_loaded = False

    def _simulate_single_choice(self, ground_truth: str, options: List[str]) -> str:
        """Simulate human single choice answer"""
        acc = self.accuracy_settings['single_choice']
        if self.rng.random() < acc:
            # Correct: return ground truth
            return ground_truth
        else:
            # Wrong: return a random incorrect option
            incorrect = [opt for opt in (options or []) if opt != ground_truth]
            if incorrect:
                return self.rng.choice(incorrect)
            return ground_truth  # Only one option available

    def _simulate_multi_choice(self, ground_truth: str, options: List[str]) -> str:
        """Simulate human multi choice answer"""
        acc = self.accuracy_settings['multi_choice']
        if self.rng.random() < acc:
            return ground_truth
        else:
            # Generate a partially incorrect answer
            gt_items = set(ground_truth.split(',')) if ground_truth else set()
            all_items = set(options or ['A', 'B', 'C', 'D'])

            # Add one incorrect or remove one correct
            if self.rng.random() < 0.5 and gt_items:
                # Remove one correct item
                to_remove = self.rng.choice(list(gt_items))
                gt_items.remove(to_remove)
                incorrect = all_items - gt_items
                if incorrect:
                    gt_items.add(self.rng.choice(list(incorrect)))
            elif gt_items:
                # Add one incorrect item
                incorrect = all_items - gt_items
                if incorrect:
                    gt_items.add(self.rng.choice(list(incorrect)))

            return ','.join(sorted(gt_items)) if gt_items else ground_truth

    def _simulate_true_false(self, ground_truth: str) -> str:
        """Simulate human true/false answer"""
        acc = self.accuracy_settings['true_false']
        if self.rng.random() < acc:
            return ground_truth
        else:
            # Flip the answer
            gt_lower = ground_truth.lower()
            if gt_lower in ['true', 'yes', '是', '正确', 't']:
                return 'False'
            elif gt_lower in ['false', 'no', '否', '错误', 'f']:
                return 'True'
            return 'False'

    def _simulate_fill_blank(self, ground_truth: str) -> str:
        """Simulate human fill blank answer with small numerical error"""
        acc = self.accuracy_settings['fill_blank']
        if self.rng.random() < acc:
            # Mostly correct with small numerical error (±5% or ±0.5)
            try:
                gt_val = float(ground_truth)
                error = self.rng.uniform(-0.05, 0.05) * max(abs(gt_val), 1)
                error += self.rng.uniform(-0.5, 0.5)
                return str(round(gt_val + error, 2))
            except:
                return ground_truth
        else:
            # Larger error
            try:
                gt_val = float(ground_truth)
                error = self.rng.uniform(-0.2, 0.2) * max(abs(gt_val), 1)
                error += self.rng.uniform(-2, 2)
                return str(round(gt_val + error, 2))
            except:
                return str(self.rng.randint(0, int(float(ground_truth)) * 2 if ground_truth else 10))

    def _simulate_short_answer(self, reference_answer: str, key_points: List[str] = None) -> str:
        """Simulate human short answer with semantic similarity"""
        if reference_answer:
            # With some probability, return the reference answer
            if self.rng.random() < 0.6:
                return reference_answer

            # Otherwise, generate a semantically similar answer
            templates = [
                "视频中{}".format(reference_answer),
                "根据观察，{}".format(reference_answer),
                reference_answer + "，这是我的观察",
                "应该是{}".format(reference_answer),
            ]

            if key_points:
                templates.append(f"从视频中可以看到{key_points[0] if key_points else ''}")

            return self.rng.choice(templates)

        # Fallback: generate a reasonable answer
        fallback_answers = [
            "视频中可以看到相关动作",
            "观察到了预期的行为",
            "根据视频内容可以确认",
            "视频显示的内容与预期一致"
        ]
        return self.rng.choice(fallback_answers)

    def inference(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        fps: float = 2.0,
        video_path: Optional[str] = None,
        question_type: str = "single_choice",
        options: List[str] = None,
        ground_truth: str = None,
        reference_answer: str = None,
        key_points: List[str] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Simulate human response based on ground truth with realistic error rates.
        """
        if not self.is_loaded:
            self.load()

        # Extract question info from metadata if available
        metadata = kwargs.get('metadata', {})

        q_type = metadata.get('question_type', question_type)
        q_options = metadata.get('options', options)
        q_answer = metadata.get('ground_truth', ground_truth)
        q_reference = metadata.get('reference_answer', reference_answer)
        q_key_points = metadata.get('key_points', key_points)

        # Generate simulated answer
        if q_type == 'single_choice':
            prediction = self._simulate_single_choice(q_answer, q_options)
        elif q_type == 'multi_choice':
            prediction = self._simulate_multi_choice(q_answer, q_options)
        elif q_type == 'true_false':
            prediction = self._simulate_true_false(q_answer)
        elif q_type == 'fill_blank':
            prediction = self._simulate_fill_blank(q_answer)
        elif q_type == 'short_answer':
            prediction = self._simulate_short_answer(q_reference, q_key_points)
        else:
            prediction = q_answer or "无法确定"

        return ModelOutput(
            raw_text=prediction,
            parsed_data=None,
            metadata={
                "model": self.model_name,
                "question_type": q_type,
                "seed": self.seed,
                "simulation": True
            }
        )

    @property
    def supports_video(self) -> bool:
        return False

    @property
    def max_frames(self) -> int:
        return 0


# Register baseline models
@ModelRegistry.register("random_guess")
class RandomGuessBaseline(RandomGuessModel):
    def __init__(self, config: Dict[str, Any] = None, local_path: Optional[str] = None):
        super().__init__("random_guess", config, local_path)


@ModelRegistry.register("human_simulation")
class HumanSimulationBaseline(HumanSimulationModel):
    def __init__(self, config: Dict[str, Any] = None, local_path: Optional[str] = None):
        super().__init__("human_simulation", config, local_path)
