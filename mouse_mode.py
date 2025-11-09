# mouse_mode.py - Mouse control mode
import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, QTimer
import pyautogui
import time

mp_hands = mp.solutions.hands

class MouseMode(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mouse Mode")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.showFullScreen()

        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.gesture_cooldown = 0
        self.smoothing_factor = 0.7  # More responsive (higher = faster response)
        self.last_smooth_pos = None
        self.last_click_time = 0
        self.click_cooldown = 0.3  # Prevent spam clicks

        # Disable PyAutoGUI failsafe and speed up
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0  # Remove delays between pyautogui commands

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)  # Faster updates

        print("\n🖱️  MOUSE MODE ACTIVE")
        print("="*60)
        print("• Index finger = Move cursor")
        print("• Pinch (thumb + index) = Left click")
        print("• 🖖 4 fingers = Return to Menu")
        print("="*60 + "\n")

    def smooth_position(self, x, y):
        if self.last_smooth_pos is None:
            self.last_smooth_pos = (x, y)
            return x, y
        
        smooth_x = self.smoothing_factor * x + (1 - self.smoothing_factor) * self.last_smooth_pos[0]
        smooth_y = self.smoothing_factor * y + (1 - self.smoothing_factor) * self.last_smooth_pos[1]
        
        self.last_smooth_pos = (smooth_x, smooth_y)
        return int(smooth_x), int(smooth_y)

    def count_extended_fingers(self, landmarks, frame_shape):
        fingers = []
        
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        if thumb_tip.x < thumb_ip.x:
            fingers.append(1)
        else:
            fingers.append(0)
        
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

    def quit_mode(self):
        print("👋 Returning to menu...")
        self.cleanup()
        QApplication.quit()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm = results.multi_hand_landmarks[0].landmark

            # Check for quit gesture (4 fingers)
            extended_fingers = self.count_extended_fingers(results.multi_hand_landmarks[0], frame.shape)
            self.gesture_cooldown = max(0, self.gesture_cooldown - 1)
            
            if extended_fingers == 4 and self.gesture_cooldown == 0:
                self.gesture_cooldown = 30
                self.quit_mode()
                return

            # Index finger position
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            # Map to screen - use direct mapping for speed
            screen_w, screen_h = pyautogui.size()
            sx = int(ix * (screen_w / w))
            sy = int(iy * (screen_h / h))

            # Light smoothing for responsiveness
            smooth_x, smooth_y = self.smooth_position(sx, sy)
            
            # Move cursor directly without easing
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)

            # Check for pinch (click) with cooldown
            dist = ((ix - tx)**2 + (iy - ty)**2)**0.5
            current_time = time.time()
            if dist < 40 and (current_time - self.last_click_time) > self.click_cooldown:
                pyautogui.click()
                self.last_click_time = current_time
                print("🖱️ Click!")

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.white))
        painter.drawText(10, 30, "🖱️  Mouse Mode | Point to move | Pinch to click | 🖖 4=Menu")

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
    mode = MouseMode()
    sys.exit(app.exec_())