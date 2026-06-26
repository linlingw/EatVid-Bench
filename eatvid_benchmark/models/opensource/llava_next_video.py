"""
LLaVA-NeXT-Video-7B-hf model implementation for video understanding.

LLaVA-NeXT-Video is based on LLaVA architecture with video support.
Architecture: LlavaNextVideoForConditionalGeneration

Reference: https://modelscope.cn/models/swift/LLaVA-NeXT-Video-7B-hf
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
import av

from models.base_model import BaseVideoModel, ModelOutput, ModelRegistry


@ModelRegistry.register("llava_next_video")
class LLaVANextVideoModel(BaseVideoModel):
    """
    LLaVA-NeXT-Video-7B-hf model for video understanding.
    
    Architecture: LLaVA-based with video support.
    Uses LlavaNextVideoForConditionalGeneration from modelscope.
    
    Video processing: Sample frames uniformly, pass as video tensor to model.
    """

    def __init__(
        self,
        model_name: str = "swift/LLaVA-NeXT-Video-7B-hf",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config)
        self.model = None
        self.processor = None
        self.device = config.get('device', 'cuda') if config else 'cuda'
        self.dtype = getattr(torch, config.get('dtype', 'float16')) if config else torch.float16
        # Video config
        self.num_frames = config.get('video_nframes', 8) if config else 8  # Number of frames to sample
        self.local_path = local_path

    def _get_model_path(self) -> str:
        return self.local_path if self.local_path else self.model_name

    def load(self) -> None:
        """Load LLaVA-NeXT-Video model."""
        if self.is_loaded:
            return

        from modelscope import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

        model_path = self._get_model_path()
        print(f"Loading LLaVA-NeXT-Video model from: {model_path}")

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.processor = LlavaNextVideoProcessor.from_pretrained(model_path)

        # Fix: patch_size may be None in some versions of LlavaNextVideoProcessor.
        # Retrieve it from the model's vision_config and set it manually.
        if getattr(self.processor, 'patch_size', None) is None:
            patch_size = getattr(self.model.config, 'vision_config', None)
            if patch_size is not None:
                patch_size = getattr(patch_size, 'patch_size', None)
            if patch_size is None:
                # Fallback: LLaVA-NeXT-Video default vision encoder uses patch_size=14
                patch_size = 14
            self.processor.patch_size = patch_size

        self.is_loaded = True
        print(f"LLaVA-NeXT-Video model loaded successfully")

    def unload(self) -> None:
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
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

    def _read_video_pyav(self, container, indices):
        """
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def _sample_frames(self, video_path):
        """Sample frames from video."""
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / self.num_frames).astype(int)
        clip = self._read_video_pyav(container, indices)
        container.close()
        return clip

    def inference(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        fps: float = 2.0,
        video_path: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Run inference with LLaVA-NeXT-Video.
        """
        if not self.is_loaded:
            self.load()

        try:
            # Get frames
            if video_path is not None:
                clip = self._sample_frames(video_path)
                print(f"[LLaVA-NeXT-Video] Sampled {len(clip)} frames")
            else:
                return ModelOutput(
                    raw_text="ERROR: Video path is required for LLaVA-NeXT-Video",
                    parsed_data=None,
                    metadata={"error": "video path required"}
                )

            # Build conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video"},
                    ],
                },
            ]

            # Apply chat template
            prompt_formatted = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Process inputs
            inputs_video = self.processor(text=prompt_formatted, videos=clip, padding=True, return_tensors="pt").to(self.model.device)

            # Run inference
            with torch.no_grad():
                output = self.model.generate(**inputs_video, max_new_tokens=max_new_tokens, do_sample=False)

            # Decode output
            output_text = self.processor.decode(output[0][2:], skip_special_tokens=True).strip()

            return ModelOutput(
                raw_text=output_text,
                parsed_data=None,
                metadata={
                    "num_frames": len(clip),
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
        return self.num_frames