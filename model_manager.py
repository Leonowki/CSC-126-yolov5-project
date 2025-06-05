"""
Model management for YOLOv5 detection models
"""

import os
import torch
from typing import Optional, List


class ModelManager:
    """Manages YOLOv5 model loading and configuration"""
    
    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.model_path: str = ""
        self.confidence_threshold: float = 0.5
        self.iou_threshold: float = 0.45
    
    def load_model(self, model_path: str) -> bool:
        """
        Load YOLOv5 model from file path
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load YOLOv5 model
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model_path = model_path
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def update_confidence(self, confidence: float) -> None:
        """Update confidence threshold"""
        self.confidence_threshold = confidence
        if self.model:
            self.model.conf = confidence
    
    def update_iou(self, iou: float) -> None:
        """Update IoU threshold"""
        self.iou_threshold = iou
        if self.model:
            self.model.iou = iou
    
    def predict(self, image):
        """
        Run inference on image
        
        Args:
            image: Input image (numpy array or PIL image)
            
        Returns:
            YOLOv5 results object
        """
        if not self.model:
            raise RuntimeError("No model loaded")
        
        # Update thresholds before inference
        self.model.conf = self.confidence_threshold
        self.model.iou = self.iou_threshold
        
        return self.model(image)
    
    def get_model_name(self) -> str:
        """Get the name of the loaded model"""
        if self.model_path:
            return os.path.basename(self.model_path)
        return "No model loaded"
    
    def get_class_names(self) -> List[str]:
        """Get list of class names from the model"""
        if self.model and hasattr(self.model, 'names'):
            return list(self.model.names.values())
        return []
    
    def is_loaded(self) -> bool:
        """Check if a model is loaded"""
        return self.model is not None