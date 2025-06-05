"""
Main GUI window for the Aerial Person Detection System
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from settings import *
from model_manager import ModelManager
from detection_processor import DetectionProcessor
from video_processor import VideoProcessor


class AerialDetectionGUI:
    """Main GUI class for the Aerial Person Detection System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(APP_GEOMETRY)
        
        # Initialize components
        self.model_manager = ModelManager()
        self.detection_processor = DetectionProcessor()
        self.video_processor = VideoProcessor()
        
        # Set up video processor callbacks
        self.video_processor.set_frame_callback(self._process_video_frame)
        self.video_processor.set_playback_control_callback(self._update_play_button)
        
        # Current image for display
        self.current_image = None
        
        # Set up UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface"""
        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Create UI components
        self._create_title_frame()
        self._create_left_panel()
        self._create_right_panel()
    
    def _create_title_frame(self):
        """Create title frame"""
        title_frame = ctk.CTkFrame(self.root, height=80)
        title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=(20, 10))
        title_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ctk.CTkLabel(title_frame, text="🛩️ Aerial Person Detection System", 
                                  font=ctk.CTkFont(size=28, weight="bold"))
        title_label.pack(expand=True)
    
    def _create_left_panel(self):
        """Create left control panel"""
        left_panel = ctk.CTkScrollableFrame(self.root, width=350)
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(0, 20))
        
        self._create_model_section(left_panel)
        self._create_parameters_section(left_panel)
        self._create_input_section(left_panel)
        self._create_legend_section(left_panel)
    
    def _create_model_section(self, parent):
        """Create model configuration section"""
        model_frame = ctk.CTkFrame(parent)
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
    
    def _create_parameters_section(self, parent):
        """Create detection parameters section"""
        params_frame = ctk.CTkFrame(parent)
        params_frame.pack(fill="x", pady=(0, 15))
        
        params_title = ctk.CTkLabel(params_frame, text="⚙️ Detection Parameters", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        params_title.pack(pady=(15, 15))
        
        # Confidence threshold
        conf_label = ctk.CTkLabel(params_frame, text="Confidence Threshold:", 
                                 font=ctk.CTkFont(size=12, weight="bold"))
        conf_label.pack(anchor="w", padx=15)
        
        self.conf_var = ctk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)
        self.conf_slider = ctk.CTkSlider(params_frame, from_=MIN_THRESHOLD, to=MAX_THRESHOLD, 
                                        variable=self.conf_var, number_of_steps=THRESHOLD_STEPS,
                                        command=self.update_conf_label)
        self.conf_slider.pack(fill="x", padx=15, pady=(5, 5))
        
        self.conf_value_label = ctk.CTkLabel(params_frame, text=f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f}", 
                                            font=ctk.CTkFont(size=11))
        self.conf_value_label.pack(pady=(0, 10))
        
        # IoU threshold
        iou_label = ctk.CTkLabel(params_frame, text="IoU Threshold:", 
                                font=ctk.CTkFont(size=12, weight="bold"))
        iou_label.pack(anchor="w", padx=15)
        
        self.iou_var = ctk.DoubleVar(value=DEFAULT_IOU_THRESHOLD)
        self.iou_slider = ctk.CTkSlider(params_frame, from_=MIN_THRESHOLD, to=MAX_THRESHOLD, 
                                       variable=self.iou_var, number_of_steps=THRESHOLD_STEPS,
                                       command=self.update_iou_label)
        self.iou_slider.pack(fill="x", padx=15, pady=(5, 5))
        
        self.iou_value_label = ctk.CTkLabel(params_frame, text=f"{DEFAULT_IOU_THRESHOLD:.2f}", 
                                           font=ctk.CTkFont(size=11))
        self.iou_value_label.pack(pady=(0, 15))
    
    def _create_input_section(self, parent):
        """Create input selection section"""
        input_frame = ctk.CTkFrame(parent)
        input_frame.pack(fill="x", pady=(0, 15))
        
        input_title = ctk.CTkLabel(input_frame, text="📁 Input Selection", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        input_title.pack(pady=(15, 15))
        
        self.load_image_btn = ctk.CTkButton(input_frame, text="📷 Load Image", 
                                           command=self.load_image,
                                           font=ctk.CTkFont(size=14, weight="bold"),
                                           fg_color=COLORS['success'], 
                                           hover_color=COLORS['success_hover'])
        self.load_image_btn.pack(fill="x", padx=15, pady=(0, 10))
        
        self.load_video_btn = ctk.CTkButton(input_frame, text="🎥 Load Video", 
                                           command=self.load_video,
                                           font=ctk.CTkFont(size=14, weight="bold"),
                                           fg_color=COLORS['error'], 
                                           hover_color=COLORS['error_hover'])
        self.load_video_btn.pack(fill="x", padx=15, pady=(0, 15))
        
        self._create_video_controls(input_frame)
    
    def _create_video_controls(self, parent):
        """Create video control buttons"""
        video_controls_frame = ctk.CTkFrame(parent)
        video_controls_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        video_controls_title = ctk.CTkLabel(video_controls_frame, text="Video Controls", 
                                           font=ctk.CTkFont(size=12, weight="bold"))
        video_controls_title.pack(pady=(10, 5))
        
        controls_button_frame = ctk.CTkFrame(video_controls_frame, fg_color="transparent")
        controls_button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.play_button = ctk.CTkButton(controls_button_frame, text="▶️ Play", 
                                        command=self.toggle_video, width=80,
                                        font=ctk.CTkFont(size=12, weight="bold"),
                                        fg_color=COLORS['warning'], 
                                        hover_color=COLORS['warning_hover'])
        self.play_button.pack(side="left", padx=(0, 5))
        
        self.stop_button = ctk.CTkButton(controls_button_frame, text="⏹️ Stop", 
                                        command=self.stop_video, width=80,
                                        font=ctk.CTkFont(size=12, weight="bold"),
                                        fg_color=COLORS['neutral'], 
                                        hover_color=COLORS['neutral_hover'])
        self.stop_button.pack(side="left")
    
    def _create_legend_section(self, parent):
        """Create class legend section"""
        legend_frame = ctk.CTkFrame(parent)
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
    
    def _create_right_panel(self):
        """Create right main content panel"""
        right_panel = ctk.CTkFrame(self.root)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(0, 20))
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=1)
        
        # Main content with tabview
        self.tabview = ctk.CTkTabview(right_panel)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        self._create_detection_tab()
        self._create_results_tab()
    
    def _create_detection_tab(self):
        """Create detection display tab"""
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
    
    def _create_results_tab(self):
        """Create results display tab"""
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
    
    # Event handlers and callbacks
    def update_conf_label(self, value):
        """Update confidence threshold label"""
        self.conf_value_label.configure(text=f"{float(value):.2f}")
        self.model_manager.update_confidence(float(value))
    
    def update_iou_label(self, value):
        """Update IoU threshold label"""
        self.iou_value_label.configure(text=f"{float(value):.2f}")
        self.model_manager.update_iou(float(value))
    
    def load_model(self):
        """Load YOLOv5 model"""
        file_path = filedialog.askopenfilename(
            title="Select YOLOv5 Model File",
            filetypes=MODEL_FILETYPES
        )
        
        if file_path:
            try:
                # Show loading
                self.load_model_btn.configure(text="Loading...", state="disabled")
                self.root.update()
                
                # Load model
                if self.model_manager.load_model(file_path):
                    model_name = self.model_manager.get_model_name()
                    self.model_label.configure(text=f"✅ {model_name}", text_color=COLORS['success'])
                    
                    class_names = self.model_manager.get_class_names()
                    messagebox.showinfo("Success", f"Model loaded successfully!\n\nAvailable classes: {class_names}")
                else:
                    raise Exception("Failed to load model")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            finally:
                self.load_model_btn.configure(text="Load YOLOv5 Model", state="normal")
    
    def load_image(self):
        """Load and process image"""
        if not self.model_manager.is_loaded():
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=IMAGE_FILETYPES
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
                self._detect_and_display_image(image)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            finally:
                self.load_image_btn.configure(text="📷 Load Image", state="normal")
    
    def load_video(self):
        """Load video file"""
        if not self.model_manager.is_loaded():
            messagebox.showwarning("Warning", "Please load a model first!")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=VIDEO_FILETYPES
        )
        
        if file_path:
            try:
                # Show loading
                self.load_video_btn.configure(text="Loading...", state="disabled")
                self.root.update()
                
                if self.video_processor.load_video(file_path):
                    # Display first frame
                    first_frame = self.video_processor.get_first_frame()
                    if first_frame is not None:
                        self._detect_and_display_image(first_frame)
                    
                    messagebox.showinfo("Success", "Video loaded successfully!\nUse the video controls to play.")
                else:
                    raise Exception("Could not open video file")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video:\n{str(e)}")
            finally:
                self.load_video_btn.configure(text="🎥 Load Video", state="normal")
    
    def toggle_video(self):
        """Toggle video playback"""
        if not self.video_processor.is_video_loaded():
            messagebox.showwarning("Warning", "Please load a video first!")
            return
        
        is_playing = self.video_processor.toggle_playback()
        self.play_button.configure(text="⏸️ Pause" if is_playing else "▶️ Play")
    
    def stop_video(self):
        """Stop video playback"""
        self.video_processor.stop_playback()
        self.video_processor.reset_video()
        self.play_button.configure(text="▶️ Play")
    
    def _detect_and_display_image(self, image):
        """Run detection on image and display results"""
        try:
            # Run inference
            results = self.model_manager.predict(image)
            
            # Draw detections
            annotated_image = self.detection_processor.draw_detections(image.copy(), results)
            
            # Display image
            self._display_image(annotated_image)
            
            # Update results
            self._update_results(results)
            
            # Switch to detection tab
            self.tabview.set("🔍 Detection")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
    
    def _display_image(self, image):
        """Display image in GUI"""
        # Resize image for display
        display_image = self.detection_processor.resize_image_for_display(
            image, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT
        )
        
        # Convert to RGB and display
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        self.image_label.configure(image=image_tk, text="")
        self.image_label.image = image_tk
    
    def _update_results(self, results):
        """Update results display"""
        # Clear previous results
        self.results_textbox.delete("0.0", "end")
        
        # Generate statistics
        stats = self.detection_processor.generate_detection_stats(results)
        
        # Format and display results
        result_text = self.detection_processor.format_detection_results(results)
        self.results_textbox.insert("0.0", result_text)
        
        # Update statistics
        stats_text = self.detection_processor.format_statistics_text(stats)
        self.stats_label.configure(text=stats_text)
    
    def _process_video_frame(self, frame):
        """Process video frame (called from video processor)"""
        # Schedule detection on main thread
        self.root.after(0, lambda: self._detect_and_display_image(frame))
    
    def _update_play_button(self, is_playing):
        """Update play button state (called from video processor)"""
        # Schedule UI update on main thread
        self.root.after(0, lambda: self.play_button.configure(
            text="⏸️ Pause" if is_playing else "▶️ Play"
        ))
    
    # Properties for backward compatibility
    @property
    def cap(self):
        """Video capture object for backward compatibility"""
        return self.video_processor.cap
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'video_processor'):
            self.video_processor.cleanup()