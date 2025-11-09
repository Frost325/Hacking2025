# main.py - Main menu system that launches different modes
import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QTimer
import subprocess
import os

mp_hands = mp.solutions.hands

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Control Menu")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.showFullScreen()

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.gesture_cooldown = 0
        self.active_process = None

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

        print("\n" + "="*60)
        print("🎮 HAND CONTROL SYSTEM - MAIN MENU")
        print("="*60)
        print("Select a mode with your hand:")
        print("   1️⃣  finger  = Drawing Mode")
        print("   3️⃣  fingers = Mouse Mode")
        print("   5️⃣  fingers = Gesture Mode")
        print("   4️⃣  fingers = QUIT APPLICATION")
        print("="*60 + "\n")

    def count_extended_fingers(self, landmarks, frame_shape):
        fingers = []
        
        # Thumb
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        if thumb_tip.x < thumb_ip.x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tip = landmarks.landmark[tip_id]
            pip = landmarks.landmark[pip_id]
            if tip.y < pip.y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return sum(fingers)

    def check_mode_selection(self, landmarks, frame_shape):
        self.gesture_cooldown = max(0, self.gesture_cooldown - 1)
        
        if self.gesture_cooldown > 0:
            return None
            
        extended_fingers = self.count_extended_fingers(landmarks, frame_shape)
        
        mode = None
        if extended_fingers == 1:
            mode = "DRAWING"
        elif extended_fingers == 3:
            mode = "MOUSE"
        elif extended_fingers == 5:
            mode = "GESTURE"
        elif extended_fingers == 4:
            mode = "QUIT"
            
        if mode:
            self.gesture_cooldown = 30
            
        return mode

    def launch_mode(self, mode):
        if mode == "QUIT":
            print("👋 Quitting application...")
            self.cleanup()
            QApplication.quit()
            return
            
        print(f"\n🚀 Launching {mode} mode...")
        
        # Stop camera and timer while subprocess runs
        self.timer.stop()
        self.cap.release()
        
        # Hide main window
        self.hide()
        
        # Launch the appropriate mode script
        mode_files = {
            "DRAWING": "draw_mode.py",
            "MOUSE": "mouse_mode.py",
            "GESTURE": "gesture_mode.py"
        }
        
        if mode in mode_files:
            try:
                # Run the mode script and wait for it to complete
                subprocess.run([sys.executable, mode_files[mode]])
            except Exception as e:
                print(f"❌ Error launching {mode}: {e}")
        
        # Restart camera and timer after subprocess exits
        print(f"\n📋 {mode} mode closed. Returning to main menu...\n")
        self.cap = cv2.VideoCapture(0)
        self.timer.start(16)
        self.show()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            mode = self.check_mode_selection(results.multi_hand_landmarks[0], frame.shape)
            if mode:
                self.launch_mode(mode)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Semi-transparent background
        painter.fillRect(self.rect(), Qt.transparent)
        
        # Draw menu UI
        font = QFont('Arial', 20, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(Qt.white))
        
        # Title
        painter.drawText(50, 80, "🎮 HAND CONTROL MENU")
        
        # Instructions
        small_font = QFont('Arial', 14)
        painter.setFont(small_font)
        
        y_pos = 150
        instructions = [
            "1️⃣  1 finger  → Drawing Mode",
            "🤟 3 fingers → Mouse Mode",
            "🖐️ 5 fingers → Gesture Mode",
            "🖖 4 fingers → Quit"
        ]
        
        for instruction in instructions:
            painter.drawText(50, y_pos, instruction)
            y_pos += 40

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = MainMenu()
    sys.exit(app.exec_())