
import customtkinter as ctk
from main_window import AerialDetectionGUI


def main():
    """Main function to run the application"""
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
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