"""
Qwen2-VL and Qwen2.5-VL model implementation for video understanding.
Uses modelscope for model loading as recommended by official documentation.
"""

import sys
from pathlib import Path

# Ensure experiments directory is in path
_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent.absolute()
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import os
import torch
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np
import json
import re

from models.base_model import BaseVideoModel, ModelOutput, ModelRegistry


class QwenVLBaseModel(BaseVideoModel):
    """
    Base class for Qwen-VL series models.
    Supports both Qwen2-VL and Qwen2.5-VL variants.
    Uses modelscope for model loading.
    """

    # Subclasses should override this
    MODEL_CLASS_NAME = "Qwen2_5_VLForConditionalGeneration"

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config)
        self.model = None
        self.processor = None
        self.local_path = local_path  # Local model path (if specified)
        self.device = config.get('device', 'cuda') if config else 'cuda'
        self.dtype = getattr(torch, config.get('dtype', 'bfloat16')) if config else torch.bfloat16

    def _get_model_path(self) -> str:
        """Get model path (local path if specified, otherwise model name)."""
        return self.local_path if self.local_path else self.model_name

    def _get_model_class(self):
        """
        Get the appropriate model class for loading.
        Subclasses should override MODEL_CLASS_NAME to specify the class.
        """
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration

        class_map = {
            "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLForConditionalGeneration,
            "Qwen2VLForConditionalGeneration": Qwen2VLForConditionalGeneration,
        }

        return class_map.get(self.MODEL_CLASS_NAME, Qwen2_5_VLForConditionalGeneration)

    def load(self) -> None:
        """
        Load Qwen-VL model and processor using transformers.
        Uses the appropriate model class based on MODEL_CLASS_NAME.
        Supports multi-GPU via device_map.
        """
        if self.is_loaded:
            return

        from transformers import AutoProcessor

        model_path = self._get_model_path()
        model_class = self._get_model_class()
        print(f"Loading model from: {model_path} using {self.MODEL_CLASS_NAME}...")

        # Determine attention implementation
        use_flash = self.config.get('use_flash_attention', True) if self.config else True
        attn_impl = "flash_attention_2" if use_flash else "eager"

        # Get GPU ID if specified (for single GPU usage)
        gpu_id = self.config.get('gpu_id', None) if self.config else None

        # Configure device map for multi-GPU support
        # If gpu_id is specified, use that single GPU
        # Otherwise, use auto to distribute across available GPUs
        if gpu_id is not None:
            device_map = {"": f"cuda:{gpu_id}"}
            print(f"Using single GPU: {gpu_id}")
        else:
            # Auto-distribute model across available GPUs
            device_map = "auto"
            print(f"Using auto device map for multi-GPU")

        try:
            self.model = model_class.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"Warning: Failed to load with {attn_impl}, trying eager attention: {e}")
            self.model = model_class.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map=device_map,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )

        # Set min/max pixels for video processing to avoid token overflow
        # Default: 256*28*28 to 1280*28*28 (256-1280 tokens per frame)
        min_pixels = self.config.get('min_pixels', 256 * 28 * 28) if self.config else 256 * 28 * 28
        max_pixels = self.config.get('max_pixels', 1280 * 28 * 28) if self.config else 1280 * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        self.is_loaded = True
        print(f"Model loaded successfully on {self.device}")
    
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
    
    def inference(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        fps: float = 2.0,
        video_path: Optional[str] = None,
        use_thinking: bool = True,
        **kwargs
    ) -> ModelOutput:
        """
        Run inference on video frames or video path.

        Args:
            frames: List of PIL Images or numpy arrays (used if video_path is None)
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            fps: Frames per second (for temporal alignment)
            video_path: Optional path to video file (preferred over frames)
            use_thinking: Whether to enable thinking mode (default: True)

        Returns:
            ModelOutput with raw text and parsed data
        """
        if not self.is_loaded:
            self.load()

        import tempfile
        import os

        temp_dir = None
        num_frames = 0

        try:
            # Add thinking mode instruction if enabled
            if use_thinking:
                # For Qwen2.5-VL, use the thinking mode format
                thinking_prompt = "Let me think step by step before answering. "
                prompt = thinking_prompt + prompt

            # Build messages based on input type
            if video_path is not None:
                # Use video path directly (recommended by official docs)
                # Round fps to nearest integer to avoid floating point precision issues
                rounded_fps = round(fps)
                # print(f"[DEBUG] Rounded fps from {fps} to {rounded_fps} to avoid precision issues")
                messages = self._build_messages_from_video_path(video_path, prompt, fps)
                num_frames = -1  # Unknown, will be handled by model
            else:
                # Convert frames to temporary files (Qwen2.5-VL requires file paths)
                pil_frames = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        pil_frames.append(Image.fromarray(frame))
                    else:
                        pil_frames.append(frame)

                num_frames = len(pil_frames)

                # Save frames to temporary files
                temp_dir = tempfile.mkdtemp()
                frame_paths = []
                for i, frame in enumerate(pil_frames):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                    frame.save(frame_path, "JPEG", quality=95)
                    frame_paths.append(f"file://{frame_path}")

                # Round fps to nearest integer
                rounded_fps = round(fps)
                messages = self._build_messages_from_frame_paths(frame_paths, prompt, rounded_fps)

            # Process inputs using qwen_vl_utils
            from qwen_vl_utils import process_vision_info

            # Debug: Print video configuration
            use_nframes = self.config.get('video_use_nframes', True) if self.config else True
            nframes = self.config.get('video_nframes', 64) if self.config else 64
            target_fps = self.config.get('video_fps', 2.0) if self.config else 2.0
            # print(f"[DEBUG] Video config - use_nframes: {use_nframes}, nframes: {nframes}, fps: {target_fps}")
            # print(f"[DEBUG] Messages content: {messages[0]['content']}")

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Debug: Print tokenization details
            # print(f"[DEBUG] Template text length: {len(text)}")
            
            # For Qwen2.5-VL, use return_video_kwargs for fps alignment
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            # print(f"[DEBUG] process_vision_info succeeded")
            # print(f"[DEBUG] image_inputs: {type(image_inputs)}, video_inputs: {type(video_inputs)}")
            if video_inputs:
                if hasattr(video_inputs, 'keys'):
                    # print(f"[DEBUG] video_inputs keys: {video_inputs.keys()}")
                    # Add debug info for video features
                    if 'pixel_values' in video_inputs:
                        print(f"[DEBUG] video pixel_values shape: {video_inputs['pixel_values'].shape}")
                else:
                    # print(f"[DEBUG] video_inputs type: {type(video_inputs)}")
                    # print(f"[DEBUG] video_inputs length: {len(video_inputs)}")
                    # Check if video_inputs is a list of dicts and convert to proper format
                    if isinstance(video_inputs, list) and len(video_inputs) > 0:
                        # print(f"[DEBUG] First video_input item type: {type(video_inputs[0])}")
                        if isinstance(video_inputs[0], dict) and 'pixel_values' in video_inputs[0]:
                            # print(f"[DEBUG] Converting list of dicts to single dict format")
                            # Convert list of dicts to single dict format
                            video_inputs = video_inputs[0]
            # print(f"[DEBUG] video_kwargs (before fix): {video_kwargs}")
            
            # 修复 fps 精度问题：将 video_kwargs 中的 fps 取整，确保与 message 中的 fps 一致
            # process_vision_info 返回的 fps 是精确的实际采样 fps（如 1.99935...），
            # 传给 processor 时会影响 second_per_grid_ts 计算，可能导致 token 数量不匹配
            if 'fps' in video_kwargs:
                raw_fps = video_kwargs['fps']
                if isinstance(raw_fps, (list, tuple)):
                    # video_kwargs['fps'] = [float(round(f)) for f in raw_fps]
                    # for finetune, use the first fps
                    video_kwargs['fps'] = float(round(raw_fps[0])) if raw_fps else 2.0
                else:
                    video_kwargs['fps'] = float(round(raw_fps))
                # print(f"[DEBUG] video_kwargs (after fps fix): {video_kwargs}")

            # Debug: Check message structure again
            # print(f"[DEBUG] Message video content: {messages[0]['content'][0]}")

            # For Qwen2.5-VL, we need to handle the video inputs carefully
            # The issue might be with the way video features are processed
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs
                )
                # print(f"[DEBUG] Processor succeeded")
                # print(f"[DEBUG] Inputs keys: {inputs.keys()}")
                # if 'pixel_values_videos' in inputs:
                #     print(f"[DEBUG] pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
                # if 'input_ids' in inputs:
                #     print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")
                #     print(f"[DEBUG] input_ids length: {len(inputs['input_ids'][0])}")
                if 'video_grid_thw' in inputs:
                    # print(f"[DEBUG] video_grid_thw: {inputs['video_grid_thw']}")
                    # Calculate expected dimensions
                    thw = inputs['video_grid_thw'].squeeze().tolist()
                    expected_features = thw[0] * thw[1] * thw[2]
                    # print(f"[DEBUG] Expected features from video_grid_thw: {expected_features}")
                # if 'second_per_grid_ts' in inputs:
                #     print(f"[DEBUG] second_per_grid_ts: {inputs['second_per_grid_ts']}")
                inputs = inputs.to("cuda")
            except Exception as e:
                print(f"[ERROR] Processor failed: {e}")
                import traceback
                traceback.print_exc()
                # Try alternative approach: use nframes with limit
                print("[DEBUG] Trying alternative approach: using limited nframes")
                # Rebuild messages with limited nframes
                alternative_messages = self._build_messages_from_video_path_alternative(video_path, prompt, fps)
                print(f"[DEBUG] Alternative message video content: {alternative_messages[0]['content'][0]}")
                # Process again
                alt_image_inputs, alt_video_inputs, alt_video_kwargs = process_vision_info(
                    alternative_messages, return_video_kwargs=True
                )
                # Handle video_inputs format for alternative approach
                if isinstance(alt_video_inputs, list) and len(alt_video_inputs) > 0:
                    if isinstance(alt_video_inputs[0], dict) and 'pixel_values' in alt_video_inputs[0]:
                        alt_video_inputs = alt_video_inputs[0]
                # 同样修复替代方案中的 fps 精度问题
                if 'fps' in alt_video_kwargs:
                    raw_fps = alt_video_kwargs['fps']
                    if isinstance(raw_fps, (list, tuple)):
                        # alt_video_kwargs['fps'] = [float(round(f)) for f in raw_fps]
                        # for finetune, use the first fps
                        alt_video_kwargs['fps'] = float(round(raw_fps[0])) if raw_fps else 2.0
                    else:
                        alt_video_kwargs['fps'] = float(round(raw_fps))
                alt_text = self.processor.apply_chat_template(
                    alternative_messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[alt_text],
                    images=alt_image_inputs,
                    videos=alt_video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **alt_video_kwargs
                )
                print(f"[DEBUG] Alternative processor succeeded")
                if 'pixel_values_videos' in inputs:
                    print(f"[DEBUG] Alternative pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
                if 'video_grid_thw' in inputs:
                    print(f"[DEBUG] Alternative video_grid_thw: {inputs['video_grid_thw']}")
                inputs = inputs.to("cuda")

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                )

            # Decode output (trim input tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Try to parse JSON from output
            parsed_data = self._try_parse_json(output_text)

            return ModelOutput(
                raw_text=output_text,
                parsed_data=parsed_data,
                metadata={
                    "num_frames": num_frames,
                    "model": self.model_name,
                    "fps": fps,
                    "video_path": video_path
                }
            )
        finally:
            # Clean up temporary files
            if temp_dir is not None:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _build_messages_from_video_path(
        self,
        video_path: str,
        prompt: str,
        fps: float = 2.0
    ) -> List[Dict]:
        """Build conversation messages from video file path.
        
        注意：nframes 和 fps 不能同时使用，否则 process_vision_info 行为不确定。
        - nframes 模式：均匀采样固定帧数，不传 fps（避免精度问题）
        - fps 模式：按指定帧率采样，fps 必须为整数以避免精度问题
        """
        # Convert to file:// URI format
        if not video_path.startswith("file://") and not video_path.startswith("http"):
            video_path = f"file://{os.path.abspath(video_path)}"

        # Get max_pixels from config to limit video resolution
        # Default: 360*420 = 151,200 pixels per frame
        max_pixels = 360 * 420
        if self.config:
            max_pixels = self.config.get('video_max_pixels', 360 * 420)

        video_content = {
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,
        }

        # Check config for which mode to use
        use_nframes = True  # Default to nframes mode
        if self.config:
            use_nframes = self.config.get('video_use_nframes', True)

        if use_nframes:
            # Use nframes: sample exactly N frames uniformly from the video
            # 注意：nframes 模式下不添加 fps 字段，避免混用导致 token 不匹配
            nframes = 64  # Default
            if self.config:
                nframes = self.config.get('video_nframes', 64)
            # For very long videos, use a reasonable limit
            nframes = min(nframes, 64)  # Maximum 64 frames
            video_content["nframes"] = nframes
            print(f"[DEBUG] Using nframes mode with {nframes} frames")
        else:
            # Use fps: sample at specified frame rate
            # 必须使用整数 fps，避免精度问题导致 token 数量不匹配
            target_fps = fps
            if self.config:
                target_fps = self.config.get('video_fps', fps)
            # 强制取整，确保 fps 是整数，避免 process_vision_info 返回的实际fps与
            # message 中的 fps 有微小差异，导致内部 token 计数不一致
            target_fps = round(target_fps)
            video_content["fps"] = target_fps
            print(f"[DEBUG] Using fps mode with {target_fps} fps")

        content = [
            video_content,
            {
                "type": "text",
                "text": prompt
            }
        ]

        return [{"role": "user", "content": content}]

    def _build_messages_from_frame_paths(
        self,
        frame_paths: List[str],
        prompt: str,
        fps: float = 2.0
    ) -> List[Dict]:
        """Build conversation messages from frame file paths."""
        # Get max_pixels from config
        max_pixels = 360 * 420
        if self.config:
            max_pixels = self.config.get('video_max_pixels', 360 * 420)

        # Limit number of frames if needed
        max_frames = 64
        if self.config:
            max_frames = self.config.get('video_nframes', 64)

        # Sample frames if we have too many
        if len(frame_paths) > max_frames:
            indices = np.linspace(0, len(frame_paths) - 1, max_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
            print(f"[DEBUG] Sampled {len(frame_paths)} frames from {len(frame_paths)} original frames")

        # For frame list, we use fps to indicate the playback speed
        # No nframes parameter needed since we already have the exact frames
        content = [
            {
                "type": "video",
                "video": frame_paths,
                "fps": fps,
                "max_pixels": max_pixels,
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

        return [{"role": "user", "content": content}]

    def _build_messages_from_video_path_alternative(
        self,
        video_path: str,
        prompt: str,
        fps: float = 2.0
    ) -> List[Dict]:
        """Alternative method to build messages (fallback when primary method fails)."""
        # Convert to file:// URI format
        if not video_path.startswith("file://") and not video_path.startswith("http"):
            video_path = f"file://{os.path.abspath(video_path)}"

        # Get max_pixels from config
        max_pixels = 360 * 420
        if self.config:
            max_pixels = self.config.get('video_max_pixels', 360 * 420)

        # Check config for which mode to use
        use_nframes = True  # Default to nframes mode
        if self.config:
            use_nframes = self.config.get('video_use_nframes', True)

        video_content = {
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,
        }

        if use_nframes:
            # Use nframes: sample exactly N frames uniformly from the video
            # 注意：nframes 模式下不添加 fps 字段，避免混用导致 token 不匹配
            nframes = 64  # Default
            if self.config:
                nframes = self.config.get('video_nframes', 64)
            # For very long videos, use a reasonable limit
            nframes = min(nframes, 64)  # Maximum 64 frames
            video_content["nframes"] = nframes
            print(f"[DEBUG] Alternative method using nframes: {nframes}")
        else:
            # Use fps: sample at specified frame rate
            # 强制取整，确保 fps 是整数，避免精度问题
            target_fps = fps
            if self.config:
                target_fps = self.config.get('video_fps', fps)
            target_fps = round(target_fps)
            video_content["fps"] = target_fps
            print(f"[DEBUG] Alternative method using fps: {target_fps}")

        content = [
            video_content,
            {
                "type": "text",
                "text": prompt
            }
        ]

        return [{"role": "user", "content": content}]
    
    def _try_parse_json(self, text: str) -> Optional[Any]:
        """Try to extract and parse JSON from model output."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON array or object
        for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    @property
    def supports_video(self) -> bool:
        return True

    @property
    def max_frames(self) -> int:
        return self.config.get('max_frames', 64) if self.config else 64


@ModelRegistry.register("qwen2_5_vl_7b")
class Qwen25VLModel(QwenVLBaseModel):
    """
    Qwen2.5-VL-7B-Instruct model for video understanding.
    This is the recommended model for eating behavior analysis.

    Uses modelscope.Qwen2_5_VLForConditionalGeneration for loading.
    """

    MODEL_CLASS_NAME = "Qwen2_5_VLForConditionalGeneration"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config, local_path)


@ModelRegistry.register("qwen2_vl_7b")
class Qwen2VLModel(QwenVLBaseModel):
    """
    Qwen2-VL-7B-Instruct model for video understanding.

    Uses modelscope.Qwen2VLForConditionalGeneration for loading.
    """

    MODEL_CLASS_NAME = "Qwen2VLForConditionalGeneration"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config, local_path)