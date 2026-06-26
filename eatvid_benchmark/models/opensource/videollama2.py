"""
VideoLLaMA2-7B model implementation for video understanding.

VideoLLaMA2-7B is a video understanding model that can process both video and image inputs.

Reference: https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B
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


@ModelRegistry.register("videollama2_7b")
class VideoLLaMA2Model(BaseVideoModel):
    """
    VideoLLaMA2-7B model for video understanding.
    
    Uses the official VideoLLaMA2 implementation from the examples directory.
    """

    def __init__(
        self,
        model_name: str = "DAMO-NLP-SG/VideoLLaMA2-7B",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config)
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.local_path = local_path
        self.device = config.get('device', 'cuda') if config else 'cuda'
        # Add examples/VideoLLaMA2-7B to path
        # videollama2_path = Path(_EXPERIMENTS_DIR) / "examples" / "VideoLLaMA2-7B"
        # if str(videollama2_path) not in sys.path:
        #     sys.path.insert(0, str(videollama2_path))

    def _get_model_path(self) -> str:
        return self.local_path if self.local_path else self.model_name

    def load(self) -> None:
        """Load VideoLLaMA2 model using the official implementation."""
        if self.is_loaded:
            return

        model_path = self._get_model_path()
        print(f"Loading VideoLLaMA2 model from: {model_path}")

        # Import VideoLLaMA2 modules
        try:
            # videollama2_path = Path(_EXPERIMENTS_DIR) / "examples" / "VideoLLaMA2-7B-16F"
            # if str(videollama2_path) not in sys.path:
            #     sys.path.insert(0, str(videollama2_path))
            
            from videollama2 import model_init
        except ImportError as e:
            print(f"Error importing VideoLLaMA2 modules: {e}")
            # Try alternative path
            # videollama2_path = Path(_EXPERIMENTS_DIR) / "examples" / "VideoLLaMA2"
            # if str(videollama2_path) not in sys.path:
            #     sys.path.insert(0, str(videollama2_path))
            try:
                from videollama2 import model_init
                print(f"Successfully imported from alternative path: {videollama2_path}")
            except ImportError as e2:
                print(f"Error importing from alternative path: {e2}")
                raise

        # Initialize model, processor, and tokenizer
        self.model, self.processor, self.tokenizer = model_init(model_path)

        self.is_loaded = True
        print(f"VideoLLaMA2 model loaded successfully")

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
        Run inference with VideoLLaMA2-7B-16F using the official implementation.
        """
        if not self.is_loaded:
            self.load()

        try:
            # Import required functions
            from videollama2 import mm_infer

            print(f"[VideoLLaMA2] Processing video: {video_path}")
            print(f"[VideoLLaMA2] Prompt: {prompt[:100]}...")

            # Process video input
            if video_path is not None:
                # Use the processor directly
                video_tensor = self.processor['video'](video_path)
                print(f"[VideoLLaMA2] Video tensor shape: {video_tensor.shape}")
            else:
                return ModelOutput(
                    raw_text="ERROR: Video path is required for VideoLLaMA2",
                    parsed_data=None,
                    metadata={"error": "video path required"}
                )

            # Execute inference
            print("[VideoLLaMA2] Running inference...")
            output = mm_infer(
                video_tensor,
                prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                modal='video',
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )

            print(f"[VideoLLaMA2] Model output: {output[:200]}...")

            # Ensure output is not empty
            if not output or output.strip() == "":
                print("[VideoLLaMA2] WARNING: Empty output from model")
                # Try with a simpler prompt
                simple_prompt = f"Please describe what you see in this video: {prompt[:50]}..."
                print(f"[VideoLLaMA2] Trying with simpler prompt: {simple_prompt[:100]}...")
                output = mm_infer(
                    video_tensor,
                    simple_prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    modal='video',
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=max_new_tokens
                )
                print(f"[VideoLLaMA2] Second attempt output: {output[:200]}...")
                if not output or output.strip() == "":
                    output = "No response generated"

            return ModelOutput(
                raw_text=output,
                parsed_data=None,
                metadata={
                    "model": self.model_name,
                    "video_path": video_path,
                    "output_length": len(output)
                }
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"ERROR: {str(e)}"
            print(f"[VideoLLaMA2] Error: {error_msg}")
            return ModelOutput(
                raw_text=error_msg,
                parsed_data=None,
                metadata={"error": str(e)}
            )

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def max_frames(self) -> int:
        # Get max frames from config or use default
        if self.config:
            return min(self.config.get('video_nframes', 32), 32)  # Maximum 32 frames
        return 32  # Default max frames for VideoLLaMA2