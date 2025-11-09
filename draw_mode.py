# draw_mode.py - Standalone drawing mode that returns to menu
import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QImage, QCursor
from PyQt5.QtCore import Qt, QTimer
import os
from datetime import datetime
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

mp_hands = mp.solutions.hands

# Load TrOCR model for handwriting recognition
print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
print("✅ TrOCR model loaded!")

class DrawingMode(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drawing Mode")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.showFullScreen()

        self.canvas = QImage(self.size(), QImage.Format_RGBA8888)
        self.canvas.fill(Qt.transparent)

        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Drawing settings
        self.smoothing_factor = 0.5
        self.prev = None
        self.drawing = False
        self.last_smooth_pos = None
        self.gesture_cooldown = 0
        
        self.brush_size = 5
        self.min_movement = 2
        self.stroke_count = 0
        self.max_strokes_before_optimize = 500

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(8)

        print("\n✍️  DRAWING MODE ACTIVE")
        print("="*60)
        print("• Pinch thumb & index to pause, unpinch to draw")
        print("• 🤟 3 fingers = Save & Run OCR")
        print("• 🖐️ 5 fingers = Clear canvas")
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

    def distance(self, pos1, pos2):
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

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

    def check_gestures(self, landmarks, frame_shape):
        extended_fingers = self.count_extended_fingers(landmarks, frame_shape)
        
        self.gesture_cooldown = max(0, self.gesture_cooldown - 1)
        
        if self.gesture_cooldown > 0:
            return None
            
        gesture = None
        
        if extended_fingers == 3:
            gesture = "save"
        elif extended_fingers == 5:
            gesture = "clear"
        elif extended_fingers == 4:
            gesture = "quit"
            
        if gesture:
            self.gesture_cooldown = 30
            
        return gesture

    def clear_canvas(self):
        self.canvas.fill(Qt.transparent)
        self.update()
        self.prev = None
        self.last_smooth_pos = None
        self.stroke_count = 0
        print("🎨 Canvas cleared!")

    def save_image(self):
        try:
            if not os.path.exists("saves"):
                os.makedirs("saves")
            filename = f"saves/drawing.png"
            
            self.canvas.save(filename, "PNG")
            
            img = cv2.imread(filename)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower1 = np.array([0, 120, 70])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 120, 70])
            upper2 = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
            
            letter = np.zeros_like(img)
            letter[mask > 0] = (255, 255, 255)
            letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
            
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(letter, kernel, iterations=3)
            final_img = cv2.bitwise_not(dilated)
            
            cv2.imwrite(filename, final_img)
            print(f"✅ Saved: {filename}")
            
            print("🔍 Running handwriting recognition...")
            image = Image.open(filename).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"📝 Recognized text: '{generated_text}'\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        self.clear_canvas()

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

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)
            dist = ((ix - tx)**2 + (iy - ty)**2)**0.5

            # Check gestures
            gesture = self.check_gestures(results.multi_hand_landmarks[0], frame.shape)
            if gesture:
                if gesture == "save":
                    self.save_image()
                elif gesture == "clear":
                    self.clear_canvas()
                elif gesture == "quit":
                    self.quit_mode()
                    return
                self.prev = None
                self.last_smooth_pos = None
                return

            # Drawing logic
            prev_drawing_state = self.drawing
            if dist >= 50:
                self.drawing = True
            elif dist <= 35:
                self.drawing = False

            if prev_drawing_state and not self.drawing:
                self.prev = None
                self.last_smooth_pos = None

            screen_w, screen_h = self.width(), self.height()
            sx = int(ix * (screen_w / w))
            sy = int(iy * (screen_h / h))

            smooth_x, smooth_y = self.smooth_position(sx, sy)
            QCursor.setPos(smooth_x, smooth_y)

            if self.drawing:
                current_pos = (smooth_x, smooth_y)
                if self.prev is not None:
                    movement = self.distance(current_pos, self.prev)
                    if movement >= self.min_movement:
                        self.draw_line(self.prev[0], self.prev[1], smooth_x, smooth_y)
                        self.prev = current_pos
                else:
                    self.prev = current_pos
            else:
                if self.prev is not None:
                    self.prev = None
        else:
            self.prev = None
            self.last_smooth_pos = None
            self.drawing = False

        self.update()

    def draw_line(self, x1, y1, x2, y2):
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        
        pen = QPen(Qt.red, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(x1, y1, x2, y2)
        painter.end()
        
        self.stroke_count += 1
        if self.stroke_count >= self.max_strokes_before_optimize:
            print("⚠️  Canvas getting heavy - consider clearing")
            self.stroke_count = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.canvas)
        
        # Display instructions
        painter.setPen(QPen(Qt.white))
        painter.drawText(10, 30, "✍️  Drawing Mode | 🤟 3=Save | 🖐️ 5=Clear | 🖖 4=Menu")

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
    mode = DrawingMode()
    sys.exit(app.exec_())