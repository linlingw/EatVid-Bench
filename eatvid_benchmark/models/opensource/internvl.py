"""
InternVL2 model implementation for video understanding.

InternVL2-8B uses InternViT-300M-448px as vision encoder + InternLM2.5-7B as LLM.
Video inference: manually sample frames, pass as list of pixel_values.
Reference: https://huggingface.co/OpenGVLab/InternVL2-8B
"""

import sys
from pathlib import Path

# Ensure experiments directory is in path
_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent.absolute()
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import os
import math
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from models.base_model import BaseVideoModel, ModelOutput, ModelRegistry


# ImageNet normalization constants (used by InternVL2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    """Build preprocessing transform for InternVL2."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 1,
                         image_size: int = 448, use_thumbnail: bool = True) -> List[Image.Image]:
    """
    Dynamic tiling for InternVL2.
    For video frames, use max_num=1 to keep one tile per frame (more efficient).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    best_ratio = (1, 1)
    best_diff = float('inf')
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))

    return processed


def _preprocess_frame(frame: Image.Image, input_size: int = 448,
                       max_num: int = 1) -> torch.Tensor:
    """
    Process a single video frame into pixel_values tensor.
    For videos, max_num=1 is recommended to limit memory usage.
    """
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(frame, image_size=input_size,
                                 use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values


@ModelRegistry.register("internvl2_8b")
class InternVL2_8BModel(BaseVideoModel):
    """
    InternVL2-8B model for video understanding.
    
    Architecture: InternViT-300M-448px + InternLM2.5-7B
    Video inference: sample frames → process each as image → pass as list
    Uses trust_remote_code=True for custom model code.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2-8B",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config)
        self.model = None
        self.tokenizer = None
        self.local_path = local_path
        self.device = config.get('device', 'cuda') if config else 'cuda'
        self.dtype = getattr(torch, config.get('dtype', 'bfloat16')) if config else torch.bfloat16
        # Video config: support both nframes mode and fps mode
        # video_use_nframes=True: sample exactly video_nframes frames uniformly
        # video_use_nframes=False: sample at video_fps rate, up to video_nframes frames max
        self.use_nframes = config.get('video_use_nframes', True) if config else True
        self.num_segments = config.get('video_nframes', 16) if config else 16
        self.target_fps = config.get('video_fps', 2.0) if config else 2.0
        self.fps_max_frames = config.get('video_nframes', 16) if config else 16  # max frames in fps mode
        self.input_size = 448  # fixed for InternVL2
        self.max_num = config.get('internvl_max_tiles', 1) if config else 1  # tiles per frame

    def _get_model_path(self) -> str:
        return self.local_path if self.local_path else self.model_name

    def load(self) -> None:
        """Load InternVL2 model and tokenizer."""
        if self.is_loaded:
            return

        from transformers import AutoModel, AutoTokenizer

        model_path = self._get_model_path()
        print(f"Loading InternVL2 model from: {model_path}")

        use_flash = self.config.get('use_flash_attention', True) if self.config else True

        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=use_flash,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
        except Exception as e:
            print(f"Warning: Failed to load with flash_attn={use_flash}, trying without: {e}")
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map="auto"
            ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.is_loaded = True
        print(f"InternVL2 model loaded successfully")

    def unload(self) -> None:
        """Unload model to free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_loaded = False
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Error during GPU memory cleanup: {e}")

    def _sample_video_frames(self, video_path: str, fps: float = None) -> List[Image.Image]:
        """
        Sample frames from video file.
        
        Supports two modes:
        - nframes mode (video_use_nframes=True): uniformly sample exactly video_nframes frames
        - fps mode (video_use_nframes=False): sample at video_fps rate, capped at video_nframes
        
        Args:
            video_path: Path to video file
            fps: Override fps for this call (only used in fps mode)
        
        Returns:
            List of PIL Images
        """
        target_fps = fps if fps is not None else self.target_fps

        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            video_fps = float(vr.get_avg_fps())
            duration = total_frames / video_fps

            if self.use_nframes:
                # nframes mode: uniform sampling of exactly num_segments frames
                num_segments = self.num_segments
                if total_frames <= num_segments:
                    frame_indices = list(range(total_frames))
                else:
                    seg_size = float(total_frames) / num_segments
                    frame_indices = [int(seg_size / 2 + seg_size * i) for i in range(num_segments)]
                print(f"[InternVL2] nframes mode: {len(frame_indices)} frames from {total_frames} total")
            else:
                # fps mode: sample at target fps, capped at max_frames
                num_frames_needed = int(duration * target_fps)
                num_frames_needed = max(1, min(num_frames_needed, self.fps_max_frames))
                if total_frames <= num_frames_needed:
                    frame_indices = list(range(total_frames))
                else:
                    seg_size = float(total_frames) / num_frames_needed
                    frame_indices = [int(seg_size / 2 + seg_size * i) for i in range(num_frames_needed)]
                print(f"[InternVL2] fps mode ({target_fps}fps, duration={duration:.1f}s): "
                      f"{len(frame_indices)} frames from {total_frames} total")

            return [Image.fromarray(vr[idx].asnumpy()).convert('RGB') for idx in frame_indices]

        except Exception as e:
            print(f"Warning: decord failed ({e}), falling back to cv2")
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = total_frames / video_fps

            if self.use_nframes:
                num_segments = self.num_segments
                if total_frames <= num_segments:
                    frame_indices = list(range(total_frames))
                else:
                    seg_size = float(total_frames) / num_segments
                    frame_indices = [int(seg_size / 2 + seg_size * i) for i in range(num_segments)]
            else:
                num_frames_needed = int(duration * target_fps)
                num_frames_needed = max(1, min(num_frames_needed, self.fps_max_frames))
                if total_frames <= num_frames_needed:
                    frame_indices = list(range(total_frames))
                else:
                    seg_size = float(total_frames) / num_frames_needed
                    frame_indices = [int(seg_size / 2 + seg_size * i) for i in range(num_frames_needed)]

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
            return frames

    def inference(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        fps: float = 2.0,
        video_path: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Run inference on video frames or video path.

        For InternVL2, video is represented as a list of frame images.
        The prompt format is: Frame1: <image>\nFrame2: <image>\n...\n{question}
        """
        if not self.is_loaded:
            self.load()

        try:
            # Get frames
            if video_path is not None:
                pil_frames = self._sample_video_frames(video_path, fps=fps)
                print(f"[InternVL2] Sampled {len(pil_frames)} frames from {video_path}")
            else:
                pil_frames = []
                for f in frames:
                    if isinstance(f, np.ndarray):
                        pil_frames.append(Image.fromarray(f))
                    else:
                        pil_frames.append(f)

            num_frames = len(pil_frames)
            if num_frames == 0:
                return ModelOutput(
                    raw_text="ERROR: No frames available",
                    parsed_data=None,
                    metadata={"error": "no frames"}
                )

            # Process each frame
            pixel_values_list = []
            num_patches_list = []
            for frame in pil_frames:
                pv = _preprocess_frame(frame, input_size=self.input_size,
                                        max_num=self.max_num)
                num_patches_list.append(pv.shape[0])
                pixel_values_list.append(pv)

            pixel_values = torch.cat(pixel_values_list).to(self.dtype).cuda()

            # Build prompt with frame references
            # Format: Frame1: <image>\nFrame2: <image>\n...\n{question}
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_frames)])
            full_question = video_prefix + prompt

            # Generation config
            generation_config = {
                'max_new_tokens': max_new_tokens,
                'do_sample': temperature > 0,
            }
            if temperature > 0:
                generation_config['temperature'] = temperature

            # Run inference
            with torch.no_grad():
                output_text = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_question,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )

            return ModelOutput(
                raw_text=output_text,
                parsed_data=None,
                metadata={
                    "num_frames": num_frames,
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
        return self.num_segments
