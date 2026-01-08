"""
Configuration Module for Buddy System
Handles all configuration settings and environment variables
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional
import cv2


@dataclass
class Config:
    """Configuration management for the Buddy system"""
    
    # Model paths
    model_path: str = "face-recog/MobileFaceNet.onnx"
    cascade_path: str = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    # Camera settings
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_buffer_size: int = 1
    
    # Recognition thresholds
    recognition_threshold: float = 0.55  # Increased to handle face variations
    confidence_threshold: float = 0.4   # Lowered confidence requirement
    min_face_size: Tuple[int, int] = (50, 50)
    max_face_size: Tuple[int, int] = (600, 600)
    
    # Timing settings
    recognition_interval: float = 2.0
    sleep_threshold: int = 30  # Frames before going to sleep
    wake_delay: float = 1.0  # Delay before greeting on wake
    frame_process_interval: float = 0.1
    
    # Motion detection
    motion_threshold: int = 30  # More lenient
    stability_frames: int = 3   # Fewer frames needed
    face_history_size: int = 10
    
    # Sleep/Wake settings
    enable_sleep_mode: bool = True
    sleep_message: str = "💤 Going to sleep... No one around."
    wake_message: str = "👀 Someone's here!"
    
    # Logging
    log_level: str = "WARNING"  # Changed from INFO to WARNING
    log_file: Optional[str] = "buddy.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        config = cls()
        config.model_path = os.getenv('BUDDY_MODEL_PATH', config.model_path)
        config.camera_index = int(os.getenv('BUDDY_CAMERA_INDEX', str(config.camera_index)))
        config.log_level = os.getenv('BUDDY_LOG_LEVEL', config.log_level)
        return config