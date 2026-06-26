"""
VideoLLaMA3-7B model implementation for video understanding.

VideoLLaMA3 uses a conversation-based interface where video is embedded 
directly in the conversation structure.
Reference: https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B
"""

import sys
import os
import gc
import warnings
from pathlib import Path

# Suppress tokenizers parallelism warning (triggered by forked processes)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Enable expandable segments to reduce CUDA memory fragmentation
# This is critical for running many consecutive inferences on the same GPU
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Ensure experiments directory is in path
_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent.absolute()
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from models.base_model import BaseVideoModel, ModelOutput, ModelRegistry


@ModelRegistry.register("videollama3_7b")
class VideoLLaMA3_7BModel(BaseVideoModel):
    """
    VideoLLaMA3-7B model for video understanding.
    
    Architecture: Qwen2.5-7B + custom SigLIP vision encoder + NaViT
    Video inference: pass video_path directly via conversation structure
    Uses trust_remote_code=True for custom model/processor code.
    
    Key feature: The processor handles all video preprocessing internally,
    just pass video_path and parameters in the conversation.
    
    Memory management: After each inference, GPU memory is explicitly freed
    to prevent CUDA OOM errors from memory fragmentation across many inferences.
    """

    def __init__(
        self,
        model_name: str = "DAMO-NLP-SG/VideoLLaMA3-7B",
        config: Dict[str, Any] = None,
        local_path: Optional[str] = None
    ):
        super().__init__(model_name, config)
        self.model = None
        self.processor = None
        self.local_path = local_path
        self.device = config.get('device', 'cuda') if config else 'cuda'
        self.dtype = getattr(torch, config.get('dtype', 'bfloat16')) if config else torch.bfloat16
        # Video config: support both nframes mode and fps mode
        # video_use_nframes=True: sample exactly video_nframes frames uniformly
        # video_use_nframes=False: sample at video_fps rate, capped at video_nframes
        self.use_nframes = config.get('video_use_nframes', True) if config else True
        self.video_fps = config.get('video_fps', 1) if config else 1
        self.video_max_frames = config.get('video_nframes', 32) if config else 32
        # GPU device index for single-card inference (avoids multi-card OOM)
        self.gpu_id = config.get('gpu_id', None) if config else None

    def _get_model_path(self) -> str:
        return self.local_path if self.local_path else self.model_name

    def _get_device_map(self):
        """
        Return device_map for model loading.
        If gpu_id is specified, load on that single GPU.
        Otherwise use "auto" (may spread across GPUs based on available memory).
        """
        if self.gpu_id is not None:
            return {"": self.gpu_id}
        return "auto"

    def load(self) -> None:
        """Load VideoLLaMA3 model and processor."""
        if self.is_loaded:
            return

        # Ensure clean memory before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = self._get_model_path()
        device_map = self._get_device_map()
        print(f"Loading VideoLLaMA3 model from: {model_path}")
        print(f"Device map: {device_map}")

        use_flash = self.config.get('use_flash_attention', True) if self.config else True

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2" if use_flash else "eager",
            )
        except Exception as e:
            print(f"Warning: Failed to load with flash_attention_2, trying eager: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=self.dtype,
                attn_implementation="eager",
            )

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.is_loaded = True
        print(f"VideoLLaMA3 model loaded successfully")

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
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: Error during GPU memory cleanup: {e}")

    def _cleanup_after_inference(self):
        """
        Clean up GPU memory after each inference to prevent fragmentation.
        Called after every generate() to avoid CUDA OOM from memory accumulation.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        Run inference using VideoLLaMA3's conversation-based interface.
        
        Video is passed as a path in the conversation structure.
        The processor handles all video preprocessing internally.
        """
        if not self.is_loaded:
            self.load()

        temp_dir = None
        try:
            if video_path is not None:
                # VideoLLaMA3 supports both fps mode and max_frames (nframes) mode
                if self.use_nframes:
                    video_config = {
                        "video_path": video_path,
                        "max_frames": self.video_max_frames
                    }
                    print(f"[VideoLLaMA3] nframes mode: max_frames={self.video_max_frames}")
                else:
                    target_fps = fps if fps is not None else self.video_fps
                    video_config = {
                        "video_path": video_path,
                        "fps": target_fps,
                        "max_frames": self.video_max_frames
                    }
                    print(f"[VideoLLaMA3] fps mode: fps={target_fps}, max_frames={self.video_max_frames}")

                content_list = [
                    {"type": "video", "video": video_config},
                    {"type": "text", "text": prompt}
                ]
            else:
                # Fall back to frames (save to temp files)
                import tempfile
                import os as _os
                temp_dir = tempfile.mkdtemp()
                frame_paths = []
                for i, frame in enumerate(frames):
                    if isinstance(frame, np.ndarray):
                        img = Image.fromarray(frame)
                    else:
                        img = frame
                    path = _os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                    img.save(path, "JPEG", quality=95)
                    frame_paths.append(path)

                content_list = []
                for fp in frame_paths:
                    content_list.append({"type": "image", "image": fp})
                content_list.append({"type": "text", "text": prompt})

            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content_list}
            ]

            # Process inputs
            inputs = self.processor(
                conversation=conversation,
                return_tensors="pt"
            )

            # Move all tensor inputs to the correct device
            # Use the model's first parameter device to ensure consistency
            target_device = next(self.model.parameters()).device
            inputs = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            input_ids_len = inputs.get("input_ids", torch.tensor([[]])).shape[-1]

            # Build generation kwargs
            # Important: when temperature=0 (greedy), do NOT pass temperature/top_p/top_k
            # to avoid UserWarning from transformers about ignored sampling params
            if temperature > 0:
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                }
            else:
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    # Explicitly suppress generation_config defaults for top_p/top_k
                    # by setting them to None (greedy decoding)
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                }

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **generate_kwargs)

            # Decode output
            # VideoLLaMA3's generate() returns the full sequence (input + generated tokens)
            # We trim the input tokens to get only the generated text.
            print(f"[VideoLLaMA3] input_ids_len={input_ids_len}, output_ids.shape={output_ids.shape}")
            if output_ids.shape[-1] > input_ids_len:
                new_token_ids = output_ids[:, input_ids_len:]
                output_text = self.processor.batch_decode(
                    new_token_ids, skip_special_tokens=True
                )[0].strip()
                print(f"[VideoLLaMA3] decoded from new_token_ids: repr={repr(output_text[:100])}")
            else:
                # output_ids.shape[-1] <= input_ids_len: VideoLLaMA3 returns only generated tokens
                output_text = self.processor.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()
                print(f"[VideoLLaMA3] decoded from full output_ids (output shorter than input): repr={repr(output_text[:100])}")
            
            # If output is still empty, try decoding full output and extracting assistant response
            if not output_text:
                print(f"[VideoLLaMA3] decoded is empty! Trying to decode full output_ids and search for assistant response")
                full_text = self.processor.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()
                print(f"[VideoLLaMA3] full output_ids decoded (first 200 chars): repr={repr(full_text[:200])}")
                # Try to extract the assistant's response from the conversation
                for marker in ["assistant\n", "assistant:", "ASSISTANT:", "<|im_start|>assistant\n"]:
                    if marker.lower() in full_text.lower():
                        idx = full_text.lower().index(marker.lower())
                        output_text = full_text[idx + len(marker):].strip()
                        print(f"[VideoLLaMA3] Extracted after '{marker}': repr={repr(output_text[:100])}")
                        break
                if not output_text:
                    output_text = full_text

            # Free memory after inference to prevent fragmentation
            del inputs, output_ids
            try:
                del new_token_ids
            except NameError:
                pass  # new_token_ids not created in the fallback path
            self._cleanup_after_inference()

            return ModelOutput(
                raw_text=output_text,
                parsed_data=None,
                metadata={
                    "num_frames": self.video_max_frames,
                    "model": self.model_name,
                    "video_path": video_path,
                    "video_fps": self.video_fps
                }
            )

        except torch.cuda.OutOfMemoryError as oom_e:
            # On OOM, clean up memory and return error
            self._cleanup_after_inference()
            import traceback
            traceback.print_exc()
            print(f"[VideoLLaMA3] CUDA OOM - consider reducing video_nframes or video_fps in config")
            return ModelOutput(
                raw_text=f"ERROR: CUDA OOM - {str(oom_e)[:200]}",
                parsed_data=None,
                metadata={"error": "oom", "message": str(oom_e)[:200]}
            )
        except Exception as e:
            self._cleanup_after_inference()
            import traceback
            traceback.print_exc()
            return ModelOutput(
                raw_text=f"ERROR: {str(e)}",
                parsed_data=None,
                metadata={"error": str(e)}
            )
        finally:
            # Clean up temp files if created
            if temp_dir is not None:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

    @property
    def supports_video(self) -> bool:
        return True

    @property
    def max_frames(self) -> int:
        return self.video_max_frames
