"""
Improved video processing utilities with non-blocking operations
"""

import cv2
import threading
import time
import queue
from typing import Callable, Optional
from settings import VIDEO_FPS


class VideoProcessor:
    """Handles video loading and playback with improved performance"""
    
    def __init__(self, process_every_nth_frame=3):
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_video_path: str = ""
        self.is_playing: bool = False
        self.video_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.frame_callback: Optional[Callable] = None
        self.playback_control_callback: Optional[Callable] = None
        
        # Performance improvements
        self.process_every_nth_frame = process_every_nth_frame
        self.frame_counter = 0
        self.frame_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.processing_active = False
        
        # Loading state
        self.is_loading = False
        self.loading_callback: Optional[Callable] = None
    
    def set_loading_callback(self, callback: Callable):
        """Set callback for loading state updates"""
        self.loading_callback = callback
    
    def load_video_async(self, video_path: str, callback: Optional[Callable] = None):
        """
        Load video file asynchronously to prevent UI freezing
        
        Args:
            video_path: Path to video file
            callback: Optional callback when loading completes
        """
        def load_worker():
            self.is_loading = True
            if self.loading_callback:
                self.loading_callback(True)
            
            success = self._load_video_sync(video_path)
            
            self.is_loading = False
            if self.loading_callback:
                self.loading_callback(False)
            
            if callback:
                callback(success)
        
        thread = threading.Thread(target=load_worker)
        thread.daemon = True
        thread.start()
    
    def _load_video_sync(self, video_path: str) -> bool:
        """
        Synchronous video loading (used internally)
        """
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                return False
            
            # Optimize video capture settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
            
            self.current_video_path = video_path
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def load_video(self, video_path: str) -> bool:
        """
        Legacy synchronous loading method (for backward compatibility)
        Consider using load_video_async() instead
        """
        return self._load_video_sync(video_path)
    
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
        """Start video playback with separate processing thread"""
        if not self.cap:
            return
        
        self.is_playing = True
        self.processing_active = True
        
        # Start video reading thread
        if not self.video_thread or not self.video_thread.is_alive():
            self.video_thread = threading.Thread(target=self._playback_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
        
        # Start frame processing thread
        if not self.processing_thread or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop_playback(self):
        """Stop video playback"""
        self.is_playing = False
        self.processing_active = False
    
    def reset_video(self):
        """Reset video to beginning"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_counter = 0
    
    def _playback_loop(self):
        """Main video playback loop (runs in separate thread)"""
        while self.is_playing and self.cap:
            ret, frame = self.cap.read()
            
            if not ret:
                # End of video
                self.is_playing = False
                self.processing_active = False
                if self.playback_control_callback:
                    self.playback_control_callback(False)
                self.reset_video()
                break
            
            self.frame_counter += 1
            
            # Only process every nth frame to reduce load
            if self.frame_counter % self.process_every_nth_frame == 0:
                try:
                    # Non-blocking queue put
                    self.frame_queue.put(frame.copy(), block=False)
                except queue.Full:
                    # Skip this frame if queue is full
                    pass
            
            # Control playback speed
            time.sleep(1/VIDEO_FPS)
    
    def _processing_loop(self):
        """Frame processing loop (runs in separate thread)"""
        while self.processing_active:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.1)
                
                # Process frame through callback
                if self.frame_callback:
                    self.frame_callback(frame)
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue
    
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
        self.processing_active = False
        
        # Wait for threads to finish
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()


# YOLOv5 optimization helper
class YOLOProcessor:
    """Helper class for optimized YOLOv5 processing"""
    
    def __init__(self, model, input_size=640, confidence_threshold=0.5):
        self.model = model
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
    
    def process_frame(self, frame):
        """
        Optimized frame processing for YOLOv5
        """
        # Resize frame for faster processing
        h, w = frame.shape[:2]
        if max(h, w) > self.input_size:
            scale = self.input_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # Run YOLOv5 inference
        results = self.model(frame)
        
        # Filter results by confidence
        detections = results.pandas().xyxy[0]
        detections = detections[detections['confidence'] > self.confidence_threshold]
        
        return detections
