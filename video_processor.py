"""
Video processing utilities
"""

import cv2
import threading
import time
from typing import Callable, Optional
from settings import VIDEO_FPS


class VideoProcessor:
    """Handles video loading and playback"""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_video_path: str = ""
        self.is_playing: bool = False
        self.video_thread: Optional[threading.Thread] = None
        self.frame_callback: Optional[Callable] = None
        self.playback_control_callback: Optional[Callable] = None
    
    def load_video(self, video_path: str) -> bool:
        """
        Load video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            bool: True if video loaded successfully
        """
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                return False
            
            self.current_video_path = video_path
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def get_first_frame(self):
        """
        Get the first frame of the video
        
        Returns:
            First frame or None if failed
        """
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            return frame
        return None
    
    def set_frame_callback(self, callback: Callable):
        """Set callback function for frame processing"""
        self.frame_callback = callback
    
    def set_playback_control_callback(self, callback: Callable):
        """Set callback function for playback control updates"""
        self.playback_control_callback = callback
    
    def toggle_playback(self) -> bool:
        """
        Toggle video playback
        
        Returns:
            bool: Current playing state
        """
        if not self.cap:
            return False
        
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
        
        return self.is_playing
    
    def start_playback(self):
        """Start video playback"""
        if not self.cap:
            return
        
        self.is_playing = True
        
        if not self.video_thread or not self.video_thread.is_alive():
            self.video_thread = threading.Thread(target=self._playback_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
    
    def stop_playback(self):
        """Stop video playback"""
        self.is_playing = False
    
    def reset_video(self):
        """Reset video to beginning"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def _playback_loop(self):
        """Main video playback loop (runs in separate thread)"""
        while self.is_playing and self.cap:
            ret, frame = self.cap.read()
            
            if not ret:
                # End of video
                self.is_playing = False
                if self.playback_control_callback:
                    self.playback_control_callback(False)
                self.reset_video()
                break
            
            # Process frame through callback
            if self.frame_callback:
                self.frame_callback(frame)
            
            # Control playback speed
            time.sleep(1/VIDEO_FPS)
    
    def is_video_loaded(self) -> bool:
        """Check if video is loaded"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_video_info(self) -> dict:
        """
        Get video information
        
        Returns:
            Dictionary with video properties
        """
        if not self.cap:
            return {}
        
        return {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.is_playing = False
        if self.cap:
            self.cap.release()