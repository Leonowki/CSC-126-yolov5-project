import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AerialDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Aerial Person Detection System")
        self.root.geometry("1400x900")
        
        # Model variables
        self.model = None
        self.model_path = ""
        self.current_image = None
        self.current_video = None
        self.video_thread = None
        self.is_playing = False
        self.cap = None
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Class colors for visualization
        self.class_colors = {
            'soldier': (0, 0, 255),      # Red
            'civilian': (0, 255, 0),     # Green
            'person': (255, 0, 0),       # Blue
            'combatant': (0, 165, 255),  # Orange
            'folks': (255, 255, 0)       # Cyan
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Title frame
        title_frame = ctk.CTkFrame(self.root, height=80)
        title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 10))
        title_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ctk.CTkLabel(title_frame, text="🛩️ Aerial Person Detection System", 
                                  font=ctk.CTkFont(size=28, weight="bold"))
        title_label.pack(expand=True)
        
        # Left panel (controls)
        left_panel = ctk.CTkScrollableFrame(self.root, width=350)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(0, 20))
        
        # Model section
        model_frame = ctk.CTkFrame(left_panel)
        model_frame.pack(fill="x", pady=(0, 15))
        
        model_title = ctk.CTkLabel(model_frame, text="🤖 Model Configuration", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        model_title.pack(pady=(15, 10))
        
        self.model_label = ctk.CTkLabel(model_frame, text="No model loaded", 
                                       font=ctk.CTkFont(size=12))
        self.model_label.pack(pady=(0, 10))
        
        self.load_model_btn = ctk.CTkButton(model_frame, text="Load YOLOv5 Model", 
                                           command=self.load_model,
                                           font=ctk.CTkFont(size=14, weight="bold"))
        self.load_model_btn.pack(pady=(0, 15))
        
        # Parameters section
        params_frame = ctk.CTkFrame(left_panel)
        params_frame.pack(fill="x", pady=(0, 15))
        
        params_title = ctk.CTkLabel(params_frame, text="⚙️ Detection Parameters", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        params_title.pack(pady=(15, 15))
        
        # Confidence threshold
        conf_label = ctk.CTkLabel(params_frame, text="Confidence Threshold:", 
                                 font=ctk.CTkFont(size=12, weight="bold"))
        conf_label.pack(anchor="w", padx=15)
        
        self.conf_var = ctk.DoubleVar(value=0.5)
        self.conf_slider = ctk.CTkSlider(params_frame, from_=0.1, to=1.0, 
                                        variable=self.conf_var, number_of_steps=18,
                                        command=self.update_conf_label)
        self.conf_slider.pack(fill="x", padx=15, pady=(5, 5))
        
        self.conf_value_label = ctk.CTkLabel(params_frame, text="0.50", 
                                            font=ctk.CTkFont(size=11))
        self.conf_value_label.pack(pady=(0, 10))
        
        # IoU threshold
        iou_label = ctk.CTkLabel(params_frame, text="IoU Threshold:", 
                                font=ctk.CTkFont(size=12, weight="bold"))
        iou_label.pack(anchor="w", padx=15)
        
        self.iou_var = ctk.DoubleVar(value=0.45)
        self.iou_slider = ctk.CTkSlider(params_frame, from_=0.1, to=1.0, 
                                       variable=self.iou_var, number_of_steps=18,
                                       command=self.update_iou_label)
        self.iou_slider.pack(fill="x", padx=15, pady=(5, 5))
        
        self.iou_value_label = ctk.CTkLabel(params_frame, text="0.45", 
                                           font=ctk.CTkFont(size=11))
        self.iou_value_label.pack(pady=(0, 15))
        
        # Input section
        input_frame = ctk.CTkFrame(left_panel)
        input_frame.pack(fill="x", pady=(0, 15))
        
        input_title = ctk.CTkLabel(input_frame, text="📁 Input Selection", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        input_title.pack(pady=(15, 15))
        
        self.load_image_btn = ctk.CTkButton(input_frame, text="📷 Load Image", 
                                           command=self.load_image,
                                           font=ctk.CTkFont(size=14, weight="bold"),
                                           fg_color="#27ae60", hover_color="#229954")
        self.load_image_btn.pack(fill="x", padx=15, pady=(0, 10))
        
        self.load_video_btn = ctk.CTkButton(input_frame, text="🎥 Load Video", 
                                           command=self.load_video,
                                           font=ctk.CTkFont(size=14, weight="bold"),
                                           fg_color="#e74c3c", hover_color="#c0392b")
        self.load_video_btn.pack(fill="x", padx=15, pady=(0, 15))
        
        # Video controls
        video_controls_frame = ctk.CTkFrame(input_frame)
        video_controls_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        video_controls_title = ctk.CTkLabel(video_controls_frame, text="Video Controls", 
                                           font=ctk.CTkFont(size=12, weight="bold"))
        video_controls_title.pack(pady=(10, 5))
        
        controls_button_frame = ctk.CTkFrame(video_controls_frame, fg_color="transparent")
        controls_button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.play_button = ctk.CTkButton(controls_button_frame, text="▶️ Play", 
                                        command=self.toggle_video, width=80,
                                        font=ctk.CTkFont(size=12, weight="bold"),
                                        fg_color="#f39c12", hover_color="#e67e22")
        self.play_button.pack(side="left", padx=(0, 5))
        
        self.stop_button = ctk.CTkButton(controls_button_frame, text="⏹️ Stop", 
                                        command=self.stop_video, width=80,
                                        font=ctk.CTkFont(size=12, weight="bold"),
                                        fg_color="#95a5a6", hover_color="#7f8c8d")
        self.stop_button.pack(side="left")
        
        # Class legend
        legend_frame = ctk.CTkFrame(left_panel)
        legend_frame.pack(fill="x", pady=(0, 15))
        
        legend_title = ctk.CTkLabel(legend_frame, text="🎨 Class Legend", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        legend_title.pack(pady=(15, 10))
        
        legend_text = """🔴 Soldier
🟢 Civilian  
🔵 Person
🟠 Combatant
🟡 Folks"""
        legend_label = ctk.CTkLabel(legend_frame, text=legend_text, 
                                   font=ctk.CTkFont(size=11), justify="left")
        legend_label.pack(pady=(0, 15))
        
        # Right panel (main content)
        right_panel = ctk.CTkFrame(self.root)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(0, 20))
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=1)
        
        # Main content with tabview
        self.tabview = ctk.CTkTabview(right_panel)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Detection tab
        detection_tab = self.tabview.add("🔍 Detection")
        detection_tab.grid_columnconfigure(0, weight=1)
        detection_tab.grid_rowconfigure(0, weight=1)
        
        # Image display frame
        self.image_frame = ctk.CTkFrame(detection_tab)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        
        self.image_label = ctk.CTkLabel(self.image_frame, 
                                       text="🖼️ Load an image or video to begin detection\n\n1. Load your YOLOv5 model\n2. Adjust detection parameters\n3. Select input file\n4. View results",
                                       font=ctk.CTkFont(size=16),
                                       wraplength=400)
        self.image_label.pack(expand=True)
        
        # Results tab
        results_tab = self.tabview.add("📊 Results")
        results_tab.grid_columnconfigure(0, weight=1)
        results_tab.grid_rowconfigure(1, weight=1)
        
        # Statistics frame
        stats_frame = ctk.CTkFrame(results_tab)
        stats_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        stats_title = ctk.CTkLabel(stats_frame, text="📈 Detection Statistics", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        stats_title.pack(pady=(15, 10))
        
        self.stats_label = ctk.CTkLabel(stats_frame, text="Statistics will appear here after detection",
                                       font=ctk.CTkFont(size=12), justify="left")
        self.stats_label.pack(pady=(0, 15))
        
        # Results text frame
        results_text_frame = ctk.CTkFrame(results_tab)
        results_text_frame.grid(row=1, column=0, sticky="nsew")
        results_text_frame.grid_columnconfigure(0, weight=1)
        results_text_frame.grid_rowconfigure(1, weight=1)
        
        results_text_title = ctk.CTkLabel(results_text_frame, text="📋 Detailed Results", 
                                         font=ctk.CTkFont(size=16, weight="bold"))
        results_text_title.grid(row=0, column=0, pady=(15, 10))
        
        self.results_textbox = ctk.CTkTextbox(results_text_frame, 
                                             font=ctk.CTkFont(family="Consolas", size=11))
        self.results_textbox.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        
    def update_conf_label(self, value):
        """Update confidence threshold label"""
        self.conf_value_label.configure(text=f"{float(value):.2f}")
    
    def update_iou_label(self, value):
        """Update IoU threshold label"""
        self.iou_value_label.configure(text=f"{float(value):.2f}")
        
    def load_model(self):
        """Load YOLOv5 model"""
        file_path = filedialog.askopenfilename(
            title="Select YOLOv5 Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Show loading
                self.load_model_btn.configure(text="Loading...", state="disabled")
                self.root.update()
                
                # Load YOLOv5 model
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=file_path)
                self.model.conf = self.conf_var.get()
                self.model.iou = self.iou_var.get()
                
                self.model_path = file_path
                model_name = os.path.basename(file_path)
                self.model_label.configure(text=f"✅ {model_name}", text_color="#27ae60")
                
                # Reset button
                self.load_model_btn.configure(text="Load YOLOv5 Model", state="normal")
                
                messagebox.showinfo("Success", f"Model loaded successfully!\n\nAvailable classes: {list(self.model.names.values())}")
                
            except Exception as e:
                self.load_model_btn.configure(text="Load YOLOv5 Model", state="normal")
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def load_image(self):
        """Load and process image"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Show loading
                self.load_image_btn.configure(text="Processing...", state="disabled")
                self.root.update()
                
                # Load and process image
                image = cv2.imread(file_path)
                self.current_image = image.copy()
                
                # Run detection
                self.detect_and_display_image(image)
                
                # Reset button
                self.load_image_btn.configure(text="📷 Load Image", state="normal")
                
            except Exception as e:
                self.load_image_btn.configure(text="📷 Load Image", state="normal")
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def load_video(self):
        """Load video file"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Show loading
                self.load_video_btn.configure(text="Loading...", state="disabled")
                self.root.update()
                
                self.current_video = file_path
                self.cap = cv2.VideoCapture(file_path)
                
                if not self.cap.isOpened():
                    raise Exception("Could not open video file")
                
                # Display first frame
                ret, frame = self.cap.read()
                if ret:
                    self.detect_and_display_image(frame)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                
                # Reset button
                self.load_video_btn.configure(text="🎥 Load Video", state="normal")
                
                messagebox.showinfo("Success", "Video loaded successfully!\nUse the video controls to play.")
                
            except Exception as e:
                self.load_video_btn.configure(text="🎥 Load Video", state="normal")
                messagebox.showerror("Error", f"Failed to load video:\n{str(e)}")
    
    def detect_and_display_image(self, image):
        """Run detection on image and display results"""
        try:
            # Update model parameters
            self.model.conf = self.conf_var.get()
            self.model.iou = self.iou_var.get()
            
            # Run inference
            results = self.model(image)
            
            # Draw detections
            annotated_image = self.draw_detections(image.copy(), results)
            
            # Display image
            self.display_image(annotated_image)
            
            # Update results
            self.update_results(results)
            
            # Switch to detection tab
            self.tabview.set("🔍 Detection")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
    
    def draw_detections(self, image, results):
        """Draw bounding boxes and labels on image"""
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Get color for class
            color = self.class_colors.get(class_name.lower(), (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background for text
            cv2.rectangle(image, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Text
            cv2.putText(image, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def display_image(self, image):
        """Display image in GUI"""
        # Resize image to fit display with reasonable defaults
        height, width = image.shape[:2]
        max_height, max_width = 600, 800
        
        if height > max_height or width > max_width:
            scale = min(max_height/height, max_width/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert to RGB and display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        self.image_label.configure(image=image_tk, text="")
        self.image_label.image = image_tk
    
    def update_results(self, results):
        """Update results display"""
        self.results_textbox.delete("0.0", "end")
        
        detections = results.pandas().xyxy[0]
        
        if len(detections) == 0:
            self.results_textbox.insert("0.0", "🚫 No detections found.\n\nTry adjusting the confidence threshold or using a different image.")
            self.stats_label.configure(text="📊 Total Detections: 0")
            return
        
        # Group by class
        class_counts = detections['name'].value_counts()
        
        self.results_textbox.insert("0.0", "🎯 DETECTION RESULTS\n" + "="*50 + "\n\n")
        
        for i, (_, detection) in enumerate(detections.iterrows(), 1):
            x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
            confidence = detection['confidence']
            class_name = detection['name']
            
            # Get emoji for class
            class_emoji = {"soldier": "🔴", "civilian": "🟢", "person": "🔵", 
                          "combatant": "🟠", "folks": "🟡"}.get(class_name.lower(), "⚪")
            
            result_text = f"Detection #{i:02d} {class_emoji}\n"
            result_text += f"├─ Class: {class_name}\n"
            result_text += f"├─ Confidence: {confidence:.3f} ({confidence*100:.1f}%)\n"
            result_text += f"└─ BBox: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})\n"
            result_text += "\n" + "─" * 40 + "\n\n"
            
            self.results_textbox.insert("end", result_text)
        
        # Update statistics
        stats_text = f"📊 Total Detections: {len(detections)}\n\n📈 Class Distribution:\n"
        for class_name, count in class_counts.items():
            emoji = {"soldier": "🔴", "civilian": "🟢", "person": "🔵", 
                    "combatant": "🟠", "folks": "🟡"}.get(class_name.lower(), "⚪")
            percentage = (count / len(detections)) * 100
            stats_text += f"{emoji} {class_name}: {count} ({percentage:.1f}%)\n"
        
        # Add confidence statistics
        avg_conf = detections['confidence'].mean()
        min_conf = detections['confidence'].min()
        max_conf = detections['confidence'].max()
        
        stats_text += f"\n🎯 Confidence Stats:\n"
        stats_text += f"   Average: {avg_conf:.3f}\n"
        stats_text += f"   Range: {min_conf:.3f} - {max_conf:.3f}"
        
        self.stats_label.configure(text=stats_text)
    
    def toggle_video(self):
        """Play/pause video"""
        if not self.current_video or not self.cap:
            messagebox.showwarning("Warning", "Please load a video first!")
            return
        
        if self.is_playing:
            self.is_playing = False
            self.play_button.configure(text="▶️ Play")
        else:
            self.is_playing = True
            self.play_button.configure(text="⏸️ Pause")
            if not self.video_thread or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self.play_video)
                self.video_thread.daemon = True
                self.video_thread.start()
    
    def stop_video(self):
        """Stop video playback"""
        self.is_playing = False
        self.play_button.configure(text="▶️ Play")
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    def play_video(self):
        """Video playback thread"""
        while self.is_playing and self.cap:
            ret, frame = self.cap.read()
            
            if not ret:
                # End of video
                self.is_playing = False
                self.root.after(0, lambda: self.play_button.configure(text="▶️ Play"))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                break
            
            # Process frame
            self.root.after(0, lambda f=frame: self.detect_and_display_image(f))
            
            # Control playback speed (approximately 30 FPS)
            time.sleep(1/30)
    
    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()

def main():
    """Main function to run the application"""
    root = ctk.CTk()
    app = AerialDetectionGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.cap:
            app.cap.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()



