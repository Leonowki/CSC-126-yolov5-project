""" Model management for YOLOv5 detection models """
import os
import torch
from typing import Optional, List


class ModelManager:
    """Manages YOLOv5 model loading and configuration"""
    
    def __init__(self, auto_load: bool = True):
        self.model: Optional[torch.nn.Module] = None
        self.model_path: str = ""
        self.confidence_threshold: float = 0.5
        self.iou_threshold: float = 0.45
        
        # Auto-load best.pt if it exists and auto_load is True
        if auto_load:
            self._auto_load_best_model()
    
    def _auto_load_best_model(self) -> None:
        """Automatically load best.pt if it exists in common locations"""
        possible_paths = [
            "best.pt",
            "./best.pt",
            "./runs/train/exp/weights/best.pt",
            "./runs/train/exp2/weights/best.pt",
            "./runs/train/exp3/weights/best.pt",
            "./weights/best.pt",
            "../best.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Auto-loading model from: {path}")
                self.load_model(path)
                break
        else:
            print("No best.pt found in common locations. Use load_model() to specify path.")

    def load_model(self, model_path: str = "best.pt") -> bool:
        """
        Load YOLOv5 model from file path
        
        Args:
            model_path: Path to the model file (defaults to "best.pt")
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load YOLOv5 model
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model_path = model_path
            print(f"Model loaded successfully: {model_path}")
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