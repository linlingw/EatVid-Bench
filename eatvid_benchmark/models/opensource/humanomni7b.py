"""
HumanOmni-7B-Video model implementation for video understanding.

HumanOmni-7B-Video is based on Qwen2.5-7B + SigLIP vision encoder (LLaVA-style).
Architecture: HumanOmniQwen2ForCausalLM (custom, requires trust_remote_code=True)
Config: model_type=HumanOmni_qwen2, num_frames=16, mm_vision_tower=google/siglip-so400m-patch14-384

Reference: https://modelscope.cn/models/myroot/HumanOmni-7B-Video
"""

import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent.absolute()
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from models.base_model import BaseVideoModel, ModelOutput, ModelRegistry


@ModelRegistry.register("humanomni_7b")
class HumanOmni7BModel(BaseVideoModel):
    """
    HumanOmni-7B-Video model for video understanding.
    
    Uses the official HumanOmni implementation from the examples directory.
    """

    def __init__(
        self,
        model_name: str = "myroot/HumanOmni-7B-Video",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.bert_tokenizer = None
        self.local_path = local_path
        self.device = config.get('device', 'cuda') if config else 'cuda'
        # Add examples/HumanOmni to path
        humanomni_path = Path(_EXPERIMENTS_DIR) / "examples" / "HumanOmni"
        if str(humanomni_path) not in sys.path:
            sys.path.insert(0, str(humanomni_path))

    def _get_model_path(self) -> str:
        return self.local_path if self.local_path else self.model_name

    def load(self) -> None:
        """Load HumanOmni model using the official implementation."""
        if self.is_loaded:
            return

        model_path = self._get_model_path()
        print(f"Loading HumanOmni model from: {model_path}")

        # Import HumanOmni modules
        try:
            from humanomni import model_init
            from transformers import BertTokenizer
        except ImportError as e:
            print(f"Error importing HumanOmni modules: {e}")
            raise

        # Initialize BERT tokenizer
        bert_model = "bert-base-uncased"
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

        # Initialize model, processor, and tokenizer
        self.model, self.processor, self.tokenizer = model_init(model_path)

        self.is_loaded = True
        print(f"HumanOmni model loaded successfully")

    def unload(self) -> None:
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.bert_tokenizer is not None:
            del self.bert_tokenizer
            self.bert_tokenizer = None
        self.is_loaded = False
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Error during GPU memory cleanup: {e}")

    def inference(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        fps: float = 2.0,
        video_path: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Run inference with HumanOmni-7B-Video using the official implementation.
        """
        if not self.is_loaded:
            self.load()

        try:
            # Import mm_infer function
            from humanomni import mm_infer

            # Process video input
            if video_path is not None:
                video_tensor = self.processor['video'](video_path)
                print(f"[HumanOmni] Processed video: {video_path}")
            else:
                return ModelOutput(
                    raw_text="ERROR: Video path is required for HumanOmni",
                    parsed_data=None,
                    metadata={"error": "video path required"}
                )

            # 执行推理，确保question参数正确传递
            # 截断prompt以避免BERT位置嵌入溢出
            # BERT-base-uncased的最大长度是512，包括特殊标记，所以我们需要更保守地截断
            truncated_prompt = prompt[:300]  # 确保不超过BERT的最大长度限制

            # Execute inference without question parameter to avoid BERT errors
            output = mm_infer(
                video_tensor,
                truncated_prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                modal='video',
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )

            return ModelOutput(
                raw_text=output,
                parsed_data=None,
                metadata={
                    "model": self.model_name,
                    "video_path": video_path
                }
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ModelOutput(
                raw_text=f"ERROR: {str(e)}",
                parsed_data=None,
                metadata={"error": str(e)}
            )

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def max_frames(self) -> int:
        # HumanOmni handles frame sampling internally
        return 64  # Default value used in the model