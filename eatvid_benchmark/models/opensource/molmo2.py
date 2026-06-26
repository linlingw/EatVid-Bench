"""
Molmo2-8B model implementation

This module provides the implementation for the Molmo2-8B video understanding model.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch

from ..base_model import BaseVideoModel, ModelOutput

# Add parent directory to path to import Molmo2 utils
_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_EXPERIMENTS_DIR / "examples" / "Molmo2-8B"))


class Molmo2Model(BaseVideoModel):
    """
    Molmo2-8B video understanding model implementation.
    """
    
    def __init__(self, model_name: str = "allenai/Molmo2-8B", config: Dict[str, Any] = None, local_path: Optional[str] = None):
        """
        Initialize the Molmo2-8B model.
        
        Args:
            model_name: Model name or path
            config: Model configuration
            local_path: Local path to the model
        """
        super().__init__(model_name, config)
        self.local_path = local_path
        self.model = None
        self.processor = None
    
    def load(self) -> None:
        """
        Load the Molmo2-8B model using modelscope.
        """
        if self.is_loaded:
            return
        
        model_path = self._get_model_path()
        print(f"Loading Molmo2-8B model from: {model_path}")
        
        try:
            from modelscope import AutoProcessor, AutoModelForImageTextToText
            
            # Load processor without device_map
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype="auto"
            )
            
            # Load model with simple device_map
            device_map = "auto"
            
            # Load model with minimal parameters
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype="auto",
                device_map=device_map
            )
            
            self.is_loaded = True
            print("Molmo2-8B model loaded successfully")
            
        except ImportError as e:
            print(f"Error importing required modules: {e}")
            raise
        except Exception as e:
            print(f"Error loading Molmo2-8B model: {e}")
            raise
    
    def unload(self) -> None:
        """
        Unload the model to free memory.
        """
        if self.is_loaded:
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()
            self.is_loaded = False
            print("Molmo2-8B model unloaded")
    
    def _get_model_path(self) -> str:
        """
        Get the model path, using local path if provided.
        """
        if self.local_path:
            return self.local_path
        return self.model_name
    
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
        Run inference with Molmo2-8B.
        
        Args:
            frames: List of video frames (not used, video_path is required)
            prompt: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            fps: Frames per second for video processing
            video_path: Path to the video file
            **kwargs: Additional parameters
            
        Returns:
            ModelOutput with generated text
        """
        if not self.is_loaded:
            self.load()
        
        try:
            # Check if video path is provided
            if video_path is None:
                return ModelOutput(
                    raw_text="ERROR: Video path is required for Molmo2-8B",
                    metadata={"error": "video path required"}
                )
            
            print(f"[Molmo2-8B] Processing video: {video_path}")
            print(f"[Molmo2-8B] Prompt: {prompt[:100]}...")
            
            # Use moderate fps to balance memory and performance
            if fps < 0.25:
                print(f"[Molmo2-8B] Increasing fps from {fps} to 0.25 to avoid token issues")
                fps = 0.25
            elif fps > 2.0:
                print(f"[Molmo2-8B] Reducing fps from {fps} to 2.0 to save memory")
                fps = 2.0
            
            print("fps:", fps)
            
            # Use process_vision_info for video handling
            from molmo_utils import process_vision_info
            
            # Create messages with video
            messages = [
                {
                    "role": "user",
                    "content": [
                        dict(type="text", text=prompt),
                        dict(type="video", video=video_path, max_fps=fps),
                    ],
                }
            ]
            
            # Process vision info
            _, videos, video_kwargs = process_vision_info(messages)
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            
            # Apply chat template
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Process inputs - use minimal parameters
            inputs = self.processor(
                videos=videos,
                video_metadata=video_metadatas,
                text=text,
                return_tensors="pt",
                **video_kwargs,
            )
            
            print(f"[Molmo2-8B] Video processed successfully: {len(videos)} videos")
            print(f"[Molmo2-8B] Inputs shape: {inputs['input_ids'].shape}")
            
            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Clear cache before inference
            torch.cuda.empty_cache()
            
            # Generate output with simple parameters
            print("[Molmo2-8B] Running inference...")
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_new_tokens, 1024)
                )
            
            # Clear cache after inference
            torch.cuda.empty_cache()
            
            # Only get generated tokens; decode them to text
            generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            print(f"[Molmo2-8B] Model output: {generated_text[:200]}...")
            
            # Ensure output is not empty
            if not generated_text or generated_text.strip() == "":
                print("[Molmo2-8B] WARNING: Empty output from model")
                generated_text = "No response generated"
            
            return ModelOutput(
                raw_text=generated_text,
                metadata={
                    "model": self.model_name,
                    "video_path": video_path,
                    "fps": fps,
                    "output_length": len(generated_text)
                }
            )
            
        except torch.cuda.OutOfMemoryError as e:
            import traceback
            traceback.print_exc()
            error_msg = f"ERROR: CUDA out of memory. Try reducing fps or using more GPUs."
            print(f"[Molmo2-8B] Error: {error_msg}")
            # Clear cache and try again with lower fps
            torch.cuda.empty_cache()
            print("[Molmo2-8B] Retrying with lower fps (1.0)...")
            return self.inference(
                frames=frames,
                prompt=prompt,
                max_new_tokens=min(max_new_tokens, 512),
                temperature=temperature,
                fps=1.0,
                video_path=video_path,
                **kwargs
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"ERROR: {str(e)}"
            print(f"[Molmo2-8B] Error: {error_msg}")
            return ModelOutput(
                raw_text=error_msg,
                metadata={"error": str(e)}
            )
    
    @property
    def supports_video(self) -> bool:
        """
        Whether the model natively supports video input.
        """
        return True
    
    @property
    def max_frames(self) -> int:
        """
        Maximum number of frames the model can process.
        """
        return 32  # Molmo2-8B can handle up to 32 frames