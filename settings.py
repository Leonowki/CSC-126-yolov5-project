
APP_TITLE = "Aerial Person Detection System"
APP_GEOMETRY = "1400x900"

# Detection parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.45
MIN_THRESHOLD = 0.1
MAX_THRESHOLD = 1.0
THRESHOLD_STEPS = 18

# Display settings
MAX_DISPLAY_HEIGHT = 600
MAX_DISPLAY_WIDTH = 800
VIDEO_FPS = 30

# Class colors for visualization (BGR format)
CLASS_COLORS = {
    'soldier': (0, 0, 255),      # Red
    'civilian': (0, 255, 0),     # Green
    'person': (255, 0, 0),       # Blue
    'combatant': (0, 165, 255),  # Orange
    'folks': (255, 255, 0)       # Cyan
}

# Class emojis for display
CLASS_EMOJIS = {
    'soldier': '🔴',
    'civilian': '🟢', 
    'person': '🔵',
    'combatant': '🟠',
    'folks': '🟡'
}

# File type filters
MODEL_FILETYPES = [("PyTorch Model", "*.pt"), ("All Files", "*.*")]
IMAGE_FILETYPES = [("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
VIDEO_FILETYPES = [("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]

# UI Colors
COLORS = {
    'success': '#27ae60',
    'success_hover': '#229954',
    'error': '#e74c3c',
    'error_hover': '#c0392b',
    'warning': '#f39c12',
    'warning_hover': '#e67e22',
    'neutral': '#95a5a6',
    'neutral_hover': '#7f8c8d'
}