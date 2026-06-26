"""
InternVideo2.5 model implementation for video understanding.

InternVideo2.5 is based on InternVL2 architecture with video support.
Architecture: InternVideo2_5_Chat_8B (custom, requires trust_remote_code=True)

Reference: https://modelscope.cn/models/OpenGVLab/InternVideo2_5_Chat_8B
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


@ModelRegistry.register("internvideo2_5")
class InternVideo25Model(BaseVideoModel):
    """
    InternVideo2.5 model for video understanding.
    
    Architecture: InternVL2-based with video support.
    Uses trust_remote_code=True for custom model loading.
    
    Video processing: Sample frames uniformly, pass as image list to model.
    The model internally processes video via its vision encoder.
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVideo2_5_Chat_8B",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config)
        self.model = None
        self.tokenizer = None
        self.device = config.get('device', 'cuda') if config else 'cuda'
        self.dtype = getattr(torch, config.get('dtype', 'bfloat16')) if config else torch.bfloat16
        # Video config: support both nframes mode and fps mode
        self.use_nframes = config.get('video_use_nframes', True) if config else True
        self.num_frames = config.get('video_nframes', 128) if config else 128  # nframes or max_frames
        self.target_fps = config.get('video_fps', 2.0) if config else 2.0
        self.local_path = local_path

    def _get_model_path(self) -> str:
        return self.local_path if self.local_path else self.model_name

    def load(self) -> None:
        """Load InternVideo2.5 model."""
        if self.is_loaded:
            return

        from modelscope import AutoModel, AutoTokenizer

        model_path = self._get_model_path()
        print(f"Loading InternVideo2.5 model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(self.dtype)

        self.is_loaded = True
        print(f"InternVideo2.5 model loaded successfully")

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

    def _sample_frames_from_video(self, video_path: str, fps: float = None) -> List[Image.Image]:
        """
        Sample frames from video.
        
        Supports nframes mode (uniform sampling) and fps mode (time-based sampling).
        """
        target_fps = fps if fps is not None else self.target_fps

        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            video_fps = float(vr.get_avg_fps())
            duration = total_frames / video_fps

            if self.use_nframes:
                # nframes mode: uniform sampling
                num_frames_to_sample = min(self.num_frames, total_frames)
                if total_frames <= num_frames_to_sample:
                    indices = list(range(total_frames))
                else:
                    seg_size = float(total_frames) / num_frames_to_sample
                    indices = [int(seg_size / 2 + seg_size * i) for i in range(num_frames_to_sample)]
                print(f"[InternVideo2.5] nframes mode: {len(indices)} frames from {total_frames} total")
            else:
                # fps mode: sample by time
                num_frames_needed = int(duration * target_fps)
                num_frames_needed = max(1, min(num_frames_needed, self.num_frames))
                if total_frames <= num_frames_needed:
                    indices = list(range(total_frames))
                else:
                    seg_size = float(total_frames) / num_frames_needed
                    indices = [int(seg_size / 2 + seg_size * i) for i in range(num_frames_needed)]
                print(f"[InternVideo2.5] fps mode ({target_fps}fps, {duration:.1f}s): "
                      f"{len(indices)} frames from {total_frames} total")

            return [Image.fromarray(vr[i].asnumpy()).convert('RGB') for i in indices]

        except Exception as e:
            print(f"Warning: decord failed ({e}), trying cv2")
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps_cv = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = total_frames / video_fps_cv

            if self.use_nframes:
                num_frames_to_sample = min(self.num_frames, total_frames)
                if total_frames <= num_frames_to_sample:
                    indices = list(range(total_frames))
                else:
                    seg_size = float(total_frames) / num_frames_to_sample
                    indices = [int(seg_size / 2 + seg_size * i) for i in range(num_frames_to_sample)]
            else:
                num_frames_needed = int(duration * target_fps)
                num_frames_needed = max(1, min(num_frames_needed, self.num_frames))
                if total_frames <= num_frames_needed:
                    indices = list(range(total_frames))
                else:
                    seg_size = float(total_frames) / num_frames_needed
                    indices = [int(seg_size / 2 + seg_size * i) for i in range(num_frames_needed)]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()
            return frames

    def _load_video(self, video_path, num_segments=128, max_num=1, get_frame_by_duration=False):
        """Load video frames and preprocess them."""
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        
        # Image preprocessing
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), 
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), 
                T.ToTensor(), 
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
            if bound:
                start, end = bound[0], bound[1]
            else:
                start, end = -100000, 100000
            start_idx = max(first_idx, round(start * fps))
            end_idx = min(round(end * fps), max_frame)
            seg_size = float(end_idx - start_idx) / num_segments
            frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
            return frame_indices

        def get_num_frames_by_duration(duration):
                local_num_frames = 4        
                num_segments = int(duration // local_num_frames)
                if num_segments == 0:
                    num_frames = local_num_frames
                else:
                    num_frames = local_num_frames * num_segments
                
                num_frames = min(512, num_frames)
                num_frames = max(128, num_frames)

                return num_frames

        transform = build_transform(input_size=448)
        if get_frame_by_duration:
            duration = max_frame / fps
            num_segments = get_num_frames_by_duration(duration)
        frame_indices = get_index(None, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def inference(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        fps: float = 2.0,
        video_path: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Run inference with InternVideo2.5.
        """
        if not self.is_loaded:
            self.load()

        try:
            # Get frames
            if video_path is not None:
                pixel_values, num_patches_list = self._load_video(video_path, num_segments=self.num_frames, max_num=1, get_frame_by_duration=False)
                pixel_values = pixel_values.to(self.dtype).to(self.model.device)
                print(f"[InternVideo2.5] Loaded {len(num_patches_list)} frames")
            else:
                return ModelOutput(
                    raw_text="ERROR: Video path is required for InternVideo2.5",
                    parsed_data=None,
                    metadata={"error": "video path required"}
                )

            # Build prompt with frame prefix
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            full_prompt = video_prefix + prompt

            # Run inference using the chat method (as per example)
            with torch.no_grad():
                # Create generation config as a dictionary instead of GenerationConfig object
                # This avoids the item assignment error in the model's chat method
                gen_config = {
                    "do_sample": temperature > 0,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "top_p": 0.1,
                    "num_beams": 1
                }
                
                # Use chat method with all required parameters
                output, chat_history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_prompt,
                    generation_config=gen_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True
                )

            return ModelOutput(
                raw_text=output,
                parsed_data=None,
                metadata={
                    "num_frames": len(num_patches_list),
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