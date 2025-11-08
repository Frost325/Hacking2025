import tkinter as tk
import time
import threading

class HandControlMenu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Control Status")
        self.root.geometry("280x180")
        self.root.configure(bg='#2b2b2b')
        self.root.resizable(False, False)
        
        # Make window always on top and semi-transparent
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.9)
        
        # Status variables
        self.current_mode = "GESTURE"
        self.current_gesture = "None"
        self.drawing_status = "Inactive"
        self.mouse_status = "Ready"
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="🎮 Hand Control Panel", 
                        font=('Arial', 12, 'bold'), fg='white', bg='#2b2b2b')
        title.pack(pady=5)
        
        # Mode Status
        mode_frame = tk.Frame(self.root, bg='#2b2b2b')
        mode_frame.pack(fill='x', padx=10, pady=2)
        
        tk.Label(mode_frame, text="Mode:", font=('Arial', 10, 'bold'), 
                fg='white', bg='#2b2b2b', width=8, anchor='w').pack(side='left')
        
        self.mode_label = tk.Label(mode_frame, text="GESTURE", font=('Arial', 10, 'bold'), 
                                  fg='#00ff00', bg='#2b2b2b', anchor='w')
        self.mode_label.pack(side='left', fill='x')
        
        # Gesture Status
        gesture_frame = tk.Frame(self.root, bg='#2b2b2b')
        gesture_frame.pack(fill='x', padx=10, pady=2)
        
        tk.Label(gesture_frame, text="Gesture:", font=('Arial', 10), 
                fg='white', bg='#2b2b2b', width=8, anchor='w').pack(side='left')
        
        self.gesture_label = tk.Label(gesture_frame, text="None", font=('Arial', 10), 
                                     fg='#ffff00', bg='#2b2b2b', anchor='w')
        self.gesture_label.pack(side='left', fill='x')
        
        # Drawing Status
        drawing_frame = tk.Frame(self.root, bg='#2b2b2b')
        drawing_frame.pack(fill='x', padx=10, pady=2)
        
        tk.Label(drawing_frame, text="Drawing:", font=('Arial', 10), 
                fg='white', bg='#2b2b2b', width=8, anchor='w').pack(side='left')
        
        self.drawing_label = tk.Label(drawing_frame, text="Inactive", font=('Arial', 10), 
                                     fg='#ff4444', bg='#2b2b2b', anchor='w')
        self.drawing_label.pack(side='left', fill='x')
        
        # Mouse Status
        mouse_frame = tk.Frame(self.root, bg='#2b2b2b')
        mouse_frame.pack(fill='x', padx=10, pady=2)
        
        tk.Label(mouse_frame, text="Mouse:", font=('Arial', 10), 
                fg='white', bg='#2b2b2b', width=8, anchor='w').pack(side='left')
        
        self.mouse_label = tk.Label(mouse_frame, text="Ready", font=('Arial', 10), 
                                   fg='#44aaff', bg='#2b2b2b', anchor='w')
        self.mouse_label.pack(side='left', fill='x')
        
        # Quick Controls Frame
        control_frame = tk.Frame(self.root, bg='#2b2b2b')
        control_frame.pack(fill='x', padx=10, pady=10)
        
        # Mode buttons
        self.gesture_btn = tk.Button(control_frame, text="Gesture", font=('Arial', 8),
                                    command=lambda: self.set_mode("GESTURE"),
                                    bg='#00aa00', fg='white', width=8)
        self.gesture_btn.pack(side='left', padx=2)
        
        self.drawing_btn = tk.Button(control_frame, text="Drawing", font=('Arial', 8),
                                    command=lambda: self.set_mode("DRAWING"),
                                    bg='#555555', fg='white', width=8)
        self.drawing_btn.pack(side='left', padx=2)
        
        self.mouse_btn = tk.Button(control_frame, text="Mouse", font=('Arial', 8),
                                  command=lambda: self.set_mode("MOUSE"),
                                  bg='#555555', fg='white', width=8)
        self.mouse_btn.pack(side='left', padx=2)
        
        # Update button colors based on current mode
        self.update_mode_buttons()
        
    def set_mode(self, mode):
        self.current_mode = mode
        self.mode_label.config(text=mode)
        self.update_mode_buttons()
        print(f"Menu: Mode changed to {mode}")
        
    def update_mode_buttons(self):
        # Reset all buttons
        self.gesture_btn.config(bg='#555555')
        self.drawing_btn.config(bg='#555555')
        self.mouse_btn.config(bg='#555555')
        
        # Highlight current mode
        if self.current_mode == "GESTURE":
            self.gesture_btn.config(bg='#00aa00')
        elif self.current_mode == "DRAWING":
            self.drawing_btn.config(bg='#00aa00')
        elif self.current_mode == "MOUSE":
            self.mouse_btn.config(bg='#00aa00')
    
    def update_gesture(self, gesture):
        self.current_gesture = gesture
        self.gesture_label.config(text=gesture)
        
    def update_drawing_status(self, status, drawing_type=None):
        self.drawing_status = status
        display_text = status
        if drawing_type:
            display_text = f"{status} ({drawing_type})"
        
        color = '#ff4444' if status == "Inactive" else '#00ff00'
        self.drawing_label.config(text=display_text, fg=color)
        
    def update_mouse_status(self, status):
        self.mouse_status = status
        color = '#44aaff' if status == "Ready" else '#ffff00'
        self.mouse_label.config(text=status, fg=color)
    
    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.root.quit()

# Simple function to run the menu
def start_control_menu():
    menu = HandControlMenu()
    menu.run()

if __name__ == "__main__":
    start_control_menu()