"""
MiniCPM-o-4_5 model implementation

This module provides the implementation for the MiniCPM-o-4_5 multimodal model.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch

from ..base_model import BaseVideoModel, ModelOutput


class MiniCPMo45Model(BaseVideoModel):
    """
    MiniCPM-o-4_5 multimodal model implementation.
    """
    
    def __init__(self, model_name: str = "OpenBMB/MiniCPM-o-4_5", config: Dict[str, Any] = None, local_path: Optional[str] = None):
        """
        Initialize the MiniCPM-o-4_5 model.
        
        Args:
            model_name: Model name or path
            config: Model configuration
            local_path: Local path to the model
        """
        super().__init__(model_name, config)
        self.local_path = local_path
        self.model = None
        self.device = config.get('device', 'cuda')
        self.dtype = config.get('dtype', torch.bfloat16)
    
    def load(self) -> None:
        """
        Load the MiniCPM-o-4_5 model using modelscope.
        """
        if self.is_loaded:
            return
        
        model_path = self._get_model_path()
        print(f"Loading MiniCPM-o-4_5 model from: {model_path}")
        
        try:
            from modelscope import AutoModel
            from minicpmo.utils import get_video_frame_audio_segments
            
            # Load model with specified parameters
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="sdpa",  # sdpa or flash_attention_2
                torch_dtype=self.dtype,
                init_vision=True,
                init_audio=True,
                init_tts=True,
            )
            
            self.model.eval().to(self.device)
            self.is_loaded = True
            print("MiniCPM-o-4_5 model loaded successfully")
            
        except ImportError as e:
            print(f"Error importing required modules: {e}")
            raise
        except Exception as e:
            print(f"Error loading MiniCPM-o-4_5 model: {e}")
            raise
    
    def unload(self) -> None:
        """
        Unload the model to free memory.
        """
        if self.is_loaded:
            self.model = None
            torch.cuda.empty_cache()
            self.is_loaded = False
            print("MiniCPM-o-4_5 model unloaded")
    
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
        video_path: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Run inference with MiniCPM-o-4_5.
        
        Args:
            frames: List of video frames (not used if video_path is provided)
            prompt: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            video_path: Path to the video file
            **kwargs: Additional parameters
            
        Returns:
            ModelOutput with generated text
        """
        if not self.is_loaded:
            self.load()
        
        try:
            from minicpmo.utils import get_video_frame_audio_segments
            
            # Get video frames
            if video_path:
                print(f"[MiniCPM-o-4_5] Processing video: {video_path}")
                video_frames, _, _ = get_video_frame_audio_segments(video_path)
                print(f"[MiniCPM-o-4_5] Extracted {len(video_frames)} frames")
            else:
                print(f"[MiniCPM-o-4_5] Using provided frames: {len(frames)}")
                video_frames = frames
            
            print(f"[MiniCPM-o-4_5] Prompt: {prompt[:100]}...")
            
            # Create messages
            msgs = [{"role": "user", "content": video_frames + [prompt]}]
            
            # Run inference
            print("[MiniCPM-o-4_5] Running inference...")
            with torch.inference_mode():
                answer = self.model.chat(
                    msgs=msgs,
                    max_new_tokens=min(max_new_tokens, 1024),
                    use_image_id=False,
                    max_slice_nums=1,
                    use_tts_template=False,
                    enable_thinking=False,
                )
            
            print(f"[MiniCPM-o-4_5] Model output: {answer[:200]}...")
            
            # Ensure output is not empty
            if not answer or answer.strip() == "":
                print("[MiniCPM-o-4_5] WARNING: Empty output from model")
                answer = "No response generated"
            
            return ModelOutput(
                raw_text=answer,
                metadata={
                    "model": self.model_name,
                    "video_path": video_path,
                    "output_length": len(answer)
                }
            )
            
        except torch.cuda.OutOfMemoryError as e:
            import traceback
            traceback.print_exc()
            error_msg = f"ERROR: CUDA out of memory."
            print(f"[MiniCPM-o-4_5] Error: {error_msg}")
            # Clear cache
            torch.cuda.empty_cache()
            return ModelOutput(
                raw_text=error_msg,
                metadata={"error": str(e)}
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"ERROR: {str(e)}"
            print(f"[MiniCPM-o-4_5] Error: {error_msg}")
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
        return 64  # MiniCPM-o-4_5 can handle up to 64 frames