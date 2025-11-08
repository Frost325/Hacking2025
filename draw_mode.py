import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtGui import QPainter, QPen, QImage, QCursor
from PyQt5.QtCore import Qt, QTimer
import os
from datetime import datetime

mp_hands = mp.solutions.hands

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Air Draw - Hand Gesture Controlled")

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.showFullScreen()

        self.canvas = QImage(self.size(), QImage.Format_RGBA8888)
        self.canvas.fill(Qt.transparent)

        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

        # Smoothing buffers
        self.position_buffer = []
        self.buffer_size = 1
        
        # Drawing settings
        self.min_distance = 15
        self.smoothing_factor = 0.7
        
        self.prev = None
        self.drawing = True
        self.last_smooth_pos = None
        self.gesture_cooldown = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def smooth_position(self, x, y):
        self.position_buffer.append((x, y))
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
        
        avg_x = sum(pos[0] for pos in self.position_buffer) / len(self.position_buffer)
        avg_y = sum(pos[1] for pos in self.position_buffer) / len(self.position_buffer)
        
        if self.last_smooth_pos:
            smooth_x = self.smoothing_factor * self.last_smooth_pos[0] + (1 - self.smoothing_factor) * avg_x
            smooth_y = self.smoothing_factor * self.last_smooth_pos[1] + (1 - self.smoothing_factor) * avg_y
        else:
            smooth_x, smooth_y = avg_x, avg_y
            
        self.last_smooth_pos = (smooth_x, smooth_y)
        return int(smooth_x), int(smooth_y)

    def distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def count_extended_fingers(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
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

    def is_fist(self, landmarks, frame_shape):
        """Check if hand is making a fist (all fingers closed)"""
        h, w = frame_shape[:2]
        
        # Check thumb - for fist, thumb should be closed
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        thumb_closed = thumb_tip.x > thumb_mcp.x  # Thumb is not extended
        
        # Check other fingers
        finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky tips
        finger_mcps = [5, 9, 13, 17]   # MCP joints (base of fingers)
        
        fingers_closed = True
        for tip_id, mcp_id in zip(finger_tips, finger_mcps):
            tip = landmarks.landmark[tip_id]
            mcp = landmarks.landmark[mcp_id]
            if tip.y < mcp.y:  # Finger tip is above MCP (finger is extended)
                fingers_closed = False
                break
        
        return thumb_closed and fingers_closed

    def check_gestures(self, landmarks, frame_shape):
        extended_fingers = self.count_extended_fingers(landmarks, frame_shape)
        is_fist_gesture = self.is_fist(landmarks, frame_shape)
        
        self.gesture_cooldown = max(0, self.gesture_cooldown - 1)
        
        if self.gesture_cooldown > 0:
            return None
            
        gesture = None
        
        # NEW: Fist gesture for clearing (much harder to trigger accidentally)
        if is_fist_gesture:
            gesture = "clear"
        # Keep the other gestures the same
        elif extended_fingers == 3:
            gesture = "save"
        elif extended_fingers == 4:
            gesture = "quit"
            
        if gesture:
            self.gesture_cooldown = 30
            
        return gesture

    def execute_gesture(self, gesture):
        if gesture == "clear":
            self.clear_canvas()
        elif gesture == "save":
            self.save_image()
        elif gesture == "quit":
            self.quit_application()

    def clear_canvas(self):
        self.canvas.fill(Qt.transparent)
        self.update()
        self.position_buffer.clear()
        self.last_smooth_pos = None
        print("🎨 Canvas cleared!")

    def save_image(self):
        try:
            if not os.path.exists("saves"):
                os.makedirs("saves")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"saves/drawing_{timestamp}.png"
            
            self.canvas.save(filename, "PNG")
            print(f"💾 Drawing saved as: {filename}")
            
        except Exception as e:
            print(f"Error saving image: {e}")

    def quit_application(self):
        reply = QMessageBox.question(self, "Quit Application", 
                                   "Are you sure you want to quit?",
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            print("👋 Application quitting...")
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

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            dist = ((ix - tx)**2 + (iy - ty)**2)**0.5

            if dist >= 45:
                self.drawing = True
            elif dist <= 35:
                self.drawing = False

            gesture = self.check_gestures(results.multi_hand_landmarks[0], frame.shape)
            if gesture:
                self.execute_gesture(gesture)

            screen_w, screen_h = self.width(), self.height()
            sx = int(ix * (screen_w / w))
            sy = int(iy * (screen_h / h))

            smooth_x, smooth_y = self.smooth_position(sx, sy)
            QCursor.setPos(smooth_x, smooth_y)

            if self.drawing:
                if self.prev is not None:
                    if self.distance(self.prev, (smooth_x, smooth_y)) >= self.min_distance:
                        self.draw_line(self.prev[0], self.prev[1], smooth_x, smooth_y)
                        self.prev = (smooth_x, smooth_y)
                else:
                    self.prev = (smooth_x, smooth_y)
            else:
                self.prev = None
                self.position_buffer.clear()
                self.last_smooth_pos = None
        else:
            self.prev = None
            self.position_buffer.clear()
            self.last_smooth_pos = None

        self.update()

    def draw_line(self, x1, y1, x2, y2):
        painter = QPainter(self.canvas)
        pen = QPen(Qt.red, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(x1, y1, x2, y2)
        painter.end()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.canvas)
        
        # Updated instructions
        painter.setPen(QPen(Qt.white))
        painter.drawText(10, 30, "✊ Fist = Clear    🤟 3 fingers = Save    🖖 4 fingers = Quit")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = Overlay()
    
    print("🎨 Air Draw - Hand Gesture Controlled")
    print("=====================================")
    print("How to use:")
    print("• Pinch thumb & index to stop drawing")
    print("• Unpinch to draw")
    print("• ✊ FIST - Clear canvas (hard to trigger accidentally)")
    print("• 🤟 3 fingers - Save drawing") 
    print("• 🖖 4 fingers - Quit application")
    print("=====================================")
    #hello
    sys.exit(app.exec_())
