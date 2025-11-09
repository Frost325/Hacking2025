# gesture_mode.py - Gesture mode for Windows shortcuts
import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QTimer
import pyautogui
import time
import os
import subprocess

mp_hands = mp.solutions.hands

class GestureMode(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Mode")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.showFullScreen()

        self.cap = cv2.VideoCapture(0)
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.gesture_cooldown = 0
        self.last_gesture_time = 0
        self.gesture_delay = 1.0  # Minimum time between gestures
        self.clap_detected = False
        self.last_clap_time = 0

        # Configure pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)

        print("\n👐  GESTURE MODE ACTIVE")
        print("="*60)
        print("Windows Shortcuts:")
        print("• ✊ Fist = Copy (Ctrl+C)")
        print("• 🖐️ 5 fingers = Paste (Ctrl+V)")
        print("• 🤟 3 fingers = Save (Ctrl+S)")
        print("• 👍 Thumbs up = Enter")
        print("• 👎 Thumbs down = File Explorer (Win+E)")
        print("• 🤙 Call me = Settings (Win+I)")
        print("• 👏 CLAP = SHUTDOWN (30s delay)")
        print("• 🖖 4 fingers = Return to Menu")
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

    def is_fist(self, landmarks, frame_shape):
        """Check if hand is making a fist (all fingers closed)"""
        fingers = []
        
        # Thumb - check if thumb is across palm
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        if thumb_tip.x > thumb_mcp.x:  # Thumb across palm
            fingers.append(0)
        else:
            fingers.append(1)
        
        # Other fingers - check if tips are below PIP joints
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tip = landmarks.landmark[tip_id]
            pip = landmarks.landmark[pip_id]
            if tip.y > pip.y:  # Finger tip below PIP joint (closed)
                fingers.append(0)
            else:
                fingers.append(1)
        
        # Fist if no fingers are extended
        return sum(fingers) == 0

    def is_thumbs_up(self, landmarks, frame_shape):
        """Check for thumbs up gesture (only thumb extended)"""
        fingers = []
        
        # Thumb should be extended
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        if thumb_tip.x < thumb_ip.x:  # Thumb extended
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers should be closed
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tip = landmarks.landmark[tip_id]
            pip = landmarks.landmark[pip_id]
            if tip.y > pip.y:  # Finger closed
                fingers.append(0)
            else:
                fingers.append(1)
        
        # Only thumb extended, others closed
        return sum(fingers) == 1 and fingers[0] == 1

    def is_thumbs_down(self, landmarks, frame_shape):
        """Check for thumbs down gesture (only thumb extended, hand rotated)"""
        extended_fingers = self.count_extended_fingers(landmarks, frame_shape)
        
        # For thumbs down, we expect only thumb extended
        if extended_fingers != 1:
            return False
            
        # Check thumb position relative to wrist
        thumb_tip = landmarks.landmark[4]
        wrist = landmarks.landmark[0]
        
        # In thumbs down, thumb is usually below wrist
        return thumb_tip.y > wrist.y

    def is_peace_sign(self, landmarks, frame_shape):
        """Check for peace sign (index and middle fingers extended)"""
        extended_fingers = self.count_extended_fingers(landmarks, frame_shape)
        
        if extended_fingers != 2:
            return False
            
        # Check which specific fingers are extended
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]
        
        # Index and middle extended, ring and pinky closed
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        ring_closed = ring_tip.y > ring_pip.y
        pinky_closed = pinky_tip.y > pinky_pip.y
        
        return index_extended and middle_extended and ring_closed and pinky_closed

    def is_call_me(self, landmarks, frame_shape):
        """Check for 'call me' gesture (thumb and pinky extended)"""
        extended_fingers = self.count_extended_fingers(landmarks, frame_shape)
        
        if extended_fingers != 2:
            return False
            
        # Check which specific fingers are extended
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        index_tip = landmarks.landmark[8]
        index_pip = landmarks.landmark[6]
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]
        
        # Thumb and pinky extended, others closed
        thumb_extended = thumb_tip.x < thumb_ip.x
        index_closed = index_tip.y > index_pip.y
        middle_closed = middle_tip.y > middle_pip.y
        ring_closed = ring_tip.y > ring_pip.y
        pinky_extended = pinky_tip.y < pinky_pip.y
        
        return thumb_extended and pinky_extended and index_closed and middle_closed and ring_closed

    def detect_clap(self, results, frame_shape):
        """Detect clapping motion with both hands"""
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
            return False
        
        # Get landmarks for both hands
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
        
        # Get palm positions (landmark 0 is wrist, but landmark 9 is palm base)
        hand1_palm = hand1.landmark[9]
        hand2_palm = hand2.landmark[9]
        
        # Calculate distance between palms
        h, w = frame_shape[:2]
        hand1_x, hand1_y = int(hand1_palm.x * w), int(hand1_palm.y * h)
        hand2_x, hand2_y = int(hand2_palm.x * w), int(hand2_palm.y * h)
        
        distance = np.sqrt((hand1_x - hand2_x)**2 + (hand1_y - hand2_y)**2)
        
        # Clap detected if palms are close together (within 100 pixels)
        clap_detected = distance < 100
        
        # Check if hands are moving toward each other (optional enhancement)
        current_time = time.time()
        
        if clap_detected and not self.clap_detected:
            # New clap detected
            self.clap_detected = True
            self.last_clap_time = current_time
            return True
        elif not clap_detected:
            self.clap_detected = False
            
        return False

    def is_open_hand(self, landmarks, frame_shape):
        """Check if hand is fully open (all fingers extended)"""
        extended_fingers = self.count_extended_fingers(landmarks, frame_shape)
        return extended_fingers == 5

    def shutdown_pc(self, method="shutdown", delay_seconds=5):
        """Shutdown PC using different methods"""
        try:
            if method == "shutdown":
                # Graceful shutdown with delay
                print(f"🔄 Shutting down in {delay_seconds} seconds...")
                print("⚠️  Press Ctrl+C in terminal to cancel!")
                if os.name == 'nt':  # Windows
                    subprocess.run(['shutdown', '/s', '/f', '/t', str(delay_seconds)])
                else:  # Linux/Mac
                    subprocess.run(['shutdown', '-h', '+1'])  # 1 minute delay
                    
            elif method == "immediate":
                # Immediate shutdown (more aggressive)
                print("🔴 IMMEDIATE SHUTDOWN!")
                if os.name == 'nt':  # Windows
                    subprocess.run(['shutdown', '/s', '/f', '/t', '0'])
                else:  # Linux/Mac
                    subprocess.run(['shutdown', '-h', 'now'])
                    
            elif method == "hibernate":
                # Hibernate
                print("💤 Hibernating PC...")
                if os.name == 'nt':  # Windows
                    subprocess.run(['shutdown', '/h'])
                else:  # Linux/Mac
                    subprocess.run(['systemctl', 'hibernate'])
                    
            elif method == "restart":
                # Restart
                print("🔄 Restarting PC...")
                if os.name == 'nt':  # Windows
                    subprocess.run(['shutdown', '/r', '/f', '/t', str(delay_seconds)])
                else:  # Linux/Mac
                    subprocess.run(['shutdown', '-r', '+1'])
                    
            return True
            
        except Exception as e:
            print(f"❌ Shutdown failed: {e}")
            return False

    def execute_shortcut(self, gesture):
        """Execute Windows shortcut based on gesture"""
        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_delay:
            return
        
        self.last_gesture_time = current_time
        
        if gesture == "fist":
            # Copy (Ctrl+C)
            pyautogui.hotkey('ctrl', 'c')
            print("📋 Copy (Ctrl+C)")
            
        elif gesture == "five_fingers":
            # Paste (Ctrl+V)
            pyautogui.hotkey('ctrl', 'v')
            print("📝 Paste (Ctrl+V)")
            
        elif gesture == "three_fingers":
            # Save (Ctrl+S)
            pyautogui.hotkey('ctrl', 's')
            print("💾 Save (Ctrl+S)")
            
        elif gesture == "thumbs_up":
            # Enter
            pyautogui.press('enter')
            print("↵ Enter")
            
        elif gesture == "thumbs_down":
            # File Explorer (Win+E)
            pyautogui.hotkey('win', 'e')
            print("📁 File Explorer (Win+E)")
            
        elif gesture == "call_me":
            # Settings (Win+I)
            pyautogui.hotkey('win', 'i')
            print("⚙️ Settings (Win+I)")
            
        elif gesture == "clap":
            # SHUTDOWN with 30 second delay
            print("👏 CLAP DETECTED - SHUTDOWN INITIATED!")
            print("⚠️  PC will shutdown in 30 seconds!")
            print("⚠️  To cancel: Open terminal and type: shutdown /a")
            
            # Show countdown
            for i in range(5, 0, -1):
                print(f"⏰ Shutdown in {i}...")
                time.sleep(1)
            
            # Execute shutdown
            success = self.shutdown_pc("shutdown", 5)
            if success:
                print("✅ Shutdown command sent successfully!")
            else:
                print("❌ Failed to send shutdown command")
            
        elif gesture == "four_fingers":
            # Return to menu
            self.quit_mode()
            return

    def check_gestures(self, landmarks, frame_shape):
        """Check for single-hand gestures"""
        extended_fingers = self.count_extended_fingers(landmarks, frame_shape)
        
        self.gesture_cooldown = max(0, self.gesture_cooldown - 1)
        
        if self.gesture_cooldown > 0:
            return None
            
        gesture = None
        
        if self.is_fist(landmarks, frame_shape):
            gesture = "fist"
        elif extended_fingers == 5:
            gesture = "five_fingers"
        elif extended_fingers == 3:
            gesture = "three_fingers"
        elif self.is_thumbs_up(landmarks, frame_shape):
            gesture = "thumbs_up"
        elif self.is_thumbs_down(landmarks, frame_shape):
            gesture = "thumbs_down"
        elif self.is_call_me(landmarks, frame_shape):
            gesture = "call_me"
        elif extended_fingers == 4:
            gesture = "four_fingers"
            
        if gesture:
            self.gesture_cooldown = 30
            
        return gesture

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

        # First check for clap (requires both hands)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            if self.detect_clap(results, frame.shape):
                current_time = time.time()
                if current_time - self.last_gesture_time >= self.gesture_delay:
                    self.execute_shortcut("clap")
        
        # Then check for single-hand gestures
        elif results.multi_hand_landmarks:
            gesture = self.check_gestures(results.multi_hand_landmarks[0], frame.shape)
            if gesture:
                self.execute_shortcut(gesture)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw compact info panel with white opaque background
        panel_width = 600
        panel_height = 180
        margin = 20
        
        # Semi-transparent white background
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.setPen(QPen(QColor(0, 0, 0, 150), 2))
        painter.drawRoundedRect(margin, margin, panel_width, panel_height, 10, 10)
        
        # Title
        font = QFont('Arial', 14, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(Qt.black))
        painter.drawText(margin + 10, margin + 25, "👐 GESTURE MODE - Windows Shortcuts")
        
        # Instructions
        small_font = QFont('Arial', 11)
        painter.setFont(small_font)
        
        y_pos = margin + 50
        instructions = [
            "✊ Fist = Copy (Ctrl+C)       🖐️ 5 fingers = Paste (Ctrl+V)",
            "🤟 3 fingers = Save (Ctrl+S)   👍 Thumbs up = Enter",
            "👎 Thumbs down = File Explorer   🤙 Call me = Settings",
            "👏 CLAP = SHUTDOWN (30s)      🖖 4 fingers = Menu"
        ]
        
        for instruction in instructions:
            painter.drawText(margin + 10, y_pos, instruction)
            y_pos += 30

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.hands:
            self.hands.close()

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mode = GestureMode()
    sys.exit(app.exec_())