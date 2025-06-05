"""
Detection processing utilities
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any
from settings import CLASS_COLORS, CLASS_EMOJIS


class DetectionProcessor:
    """Handles detection processing and visualization"""
    
    def __init__(self):
        self.class_colors = CLASS_COLORS
        self.class_emojis = CLASS_EMOJIS
    
    def draw_detections(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (BGR format)
            results: YOLOv5 results object
            
        Returns:
            Annotated image with bounding boxes
        """
        annotated_image = image.copy()
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \
                            int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Get color for class
            color = self.class_colors.get(class_name.lower(), (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background for text
            cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Text
            cv2.putText(annotated_image, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_image
    
    def resize_image_for_display(self, image: np.ndarray, 
                                max_width: int = 800, 
                                max_height: int = 600) -> np.ndarray:
        """
        Resize image to fit display while maintaining aspect ratio
        
        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if height > max_height or width > max_width:
            scale = min(max_height/height, max_width/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
    
    def generate_detection_stats(self, results) -> Dict[str, Any]:
        """
        Generate detection statistics
        
        Args:
            results: YOLOv5 results object
            
        Returns:
            Dictionary containing detection statistics
        """
        detections = results.pandas().xyxy[0]
        
        if len(detections) == 0:
            return {
                'total_detections': 0,
                'class_counts': {},
                'confidence_stats': {}
            }
        
        # Class distribution
        class_counts = detections['name'].value_counts().to_dict()
        
        # Confidence statistics
        confidence_stats = {
            'average': float(detections['confidence'].mean()),
            'min': float(detections['confidence'].min()),
            'max': float(detections['confidence'].max())
        }
        
        return {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'confidence_stats': confidence_stats
        }
    
    def format_detection_results(self, results) -> str:
        """
        Format detection results as text
        
        Args:
            results: YOLOv5 results object
            
        Returns:
            Formatted text string
        """
        detections = results.pandas().xyxy[0]
        
        if len(detections) == 0:
            return "🚫 No detections found.\n\nTry adjusting the confidence threshold or using a different image."
        
        result_text = "🎯 DETECTION RESULTS\n" + "="*50 + "\n\n"
        
        for i, (_, detection) in enumerate(detections.iterrows(), 1):
            x1, y1, x2, y2 = detection['xmin'], detection['ymin'], \
                            detection['xmax'], detection['ymax']
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Get emoji for class
            class_emoji = self.class_emojis.get(class_name.lower(), "⚪")
            
            detection_text = f"Detection #{i:02d} {class_emoji}\n"
            detection_text += f"├─ Class: {class_name}\n"
            detection_text += f"├─ Confidence: {confidence:.3f} ({confidence*100:.1f}%)\n"
            detection_text += f"└─ BBox: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})\n"
            detection_text += "\n" + "─" * 40 + "\n\n"
            
            result_text += detection_text
        
        return result_text
    
    def format_statistics_text(self, stats: Dict[str, Any]) -> str:
        """
        Format statistics as display text
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Formatted statistics text
        """
        if stats['total_detections'] == 0:
            return "📊 Total Detections: 0"
        
        stats_text = f"📊 Total Detections: {stats['total_detections']}\n\n📈 Class Distribution:\n"
        
        for class_name, count in stats['class_counts'].items():
            emoji = self.class_emojis.get(class_name.lower(), "⚪")
            percentage = (count / stats['total_detections']) * 100
            stats_text += f"{emoji} {class_name}: {count} ({percentage:.1f}%)\n"
        
        # Add confidence statistics
        conf_stats = stats['confidence_stats']
        stats_text += f"\n🎯 Confidence Stats:\n"
        stats_text += f"   Average: {conf_stats['average']:.3f}\n"
        stats_text += f"   Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}"
        
        return stats_text