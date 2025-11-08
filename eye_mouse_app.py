import sys
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard
import time
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt6.QtCore import QThread, pyqtSignal

SMOOTHING_ALPHA = 0.25  # for optional EMA smoothing on top of Kalman

# --- Kalman filter class for smoothing ---
class KalmanFilter1D:
    def __init__(self, process_var=1e-3, meas_var=1e-2):
        self.process_var = process_var
        self.meas_var = meas_var
        self.x = None
        self.P = None

    def update(self, z):
        if self.x is None:
            self.x = z
            self.P = 1.0
            return self.x
        self.P += self.process_var
        K = self.P / (self.P + self.meas_var)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x

class EyeTrackerThread(QThread):
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.tracking = False
        self.calibrated = False
        self.T = None
        self.blink_flag = False
        self.last_blink_time = 0
        self.right_click_timeout = 1.0  # seconds to detect double blink

        self.screen_w, self.screen_h = pyautogui.size()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.cap = None

        # Kalman filters for x and y coords
        # Adjusted params for better vertical smoothing
        self.kf_x = KalmanFilter1D(process_var=1e-3, meas_var=1e-2)
        self.kf_y = KalmanFilter1D(process_var=5e-3, meas_var=5e-3)

        # EMA smoothing variables
        self.ema_x = None
        self.ema_y = None

    def relative_eye_position(self, lm):
        # Dynamic normalization inside live eye bounding box for head movement robustness
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        ix = np.mean([lm[i].x for i in LEFT_IRIS + RIGHT_IRIS])
        iy = np.mean([lm[i].y for i in LEFT_IRIS + RIGHT_IRIS])

        # Use combined eye bounding box (both eyes)
        eye_left = min(lm[33].x, lm[362].x)
        eye_right = max(lm[133].x, lm[263].x)
        eye_top = min(lm[159].y, lm[386].y)
        eye_bottom = max(lm[145].y, lm[374].y)

        eye_height = max(eye_bottom - eye_top, 0.02)  # avoid too small height

        nx = (ix - eye_left) / (eye_right - eye_left + 1e-6)
        ny = (iy - eye_top) / (eye_height + 1e-6)

        # Clamp between 0 and 1
        nx = np.clip(nx, 0, 1)
        ny = np.clip(ny, 0, 1)

        return np.array([nx, ny])

    def map_to_screen(self, norm_xy):
        # Map normalized gaze position to calibrated screen coords
        v = np.array([norm_xy[0], norm_xy[1], 1.0])
        scr = v @ self.T
        scr[0] = np.clip(scr[0], 0, self.screen_w - 1)
        scr[1] = np.clip(scr[1], 0, self.screen_h - 1)
        return scr.astype(int)

    def eye_aspect_ratio(self, lm, left=True):
        # EAR formula to detect blinks
        if left:
            top = lm[159]
            bottom = lm[145]
            left_pt = lm[33]
            right_pt = lm[133]
        else:
            top = lm[386]
            bottom = lm[374]
            left_pt = lm[362]
            right_pt = lm[263]

        v = np.linalg.norm([top.x - bottom.x, top.y - bottom.y])
        h = np.linalg.norm([left_pt.x - right_pt.x, left_pt.y - right_pt.y])
        return v / (h + 1e-6)

    def calibrate(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.status.emit("Calibration: Look at each green dot and press SPACE.")

        CAP_POINTS = [
            ("top-left", (0.1, 0.1)),
            ("top-right", (0.9, 0.1)),
            ("bottom-left", (0.1, 0.9)),
            ("bottom-right", (0.9, 0.9)),
            ("center", (0.5, 0.5)),
        ]

        cam_pts, scr_pts = [], []

        # Calculate smaller head box around eyes for guidance (~40% width, 25% height)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]

            # Smaller head guidance green box centered roughly on eyes
            box_w, box_h = int(w * 0.4), int(h * 0.25)
            # Eyes tend to be around upper half, so offset box slightly upward
            top_left = (w // 2 - box_w // 2, int(h * 0.35))
            bottom_right = (top_left[0] + box_w, top_left[1] + box_h)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            for name, (nx, ny) in CAP_POINTS:
                cv2.circle(frame, (int(nx * w), int(ny * h)), 18, (0, 255, 0), 2)

            cv2.putText(frame, "Press SPACE to capture point, Q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Calibration", frame)

            cam_pts = []
            scr_pts = []

            for name, (nx, ny) in CAP_POINTS:
                self.status.emit(f"Look at {name}, then press SPACE")
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    h, w = frame.shape[:2]
                    frame2 = frame.copy()

                    cv2.rectangle(frame2, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.circle(frame2, (int(nx * w), int(ny * h)), 18, (0, 255, 0), 2)

                    cv2.putText(frame2, f"Look at {name} and press SPACE", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Calibration", frame2)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):
                        res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if res.multi_face_landmarks:
                            lm = res.multi_face_landmarks[0].landmark
                            cam_pts.append(self.relative_eye_position(lm))
                            scr_pts.append(np.array([nx * self.screen_w, ny * self.screen_h]))
                            break
                    elif key == ord('q'):
                        cv2.destroyWindow("Calibration")
                        return

            if len(cam_pts) == len(CAP_POINTS):
                break

        cv2.destroyWindow("Calibration")

        src = np.array(cam_pts)
        dst = np.array(scr_pts)
        A = np.hstack([src, np.ones((len(src), 1))])
        self.T, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)

        self.calibrated = True
        self.status.emit("Calibration complete.")

    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.status.emit("Camera failed to open.")
            return
        self.status.emit("Camera started.")

        while self.running:
            if keyboard.is_pressed('q'):
                if self.tracking:
                    self.tracking = False
                    self.status.emit("Tracking OFF (Q pressed)")

            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks and self.calibrated:
                lm = results.multi_face_landmarks[0].landmark

                # Blink detection (eye aspect ratio)
                ear_left = self.eye_aspect_ratio(lm, True)
                ear_right = self.eye_aspect_ratio(lm, False)
                ear = (ear_left + ear_right) / 2

                # Blink click logic
                if ear < 0.21:
                    if not self.blink_flag:
                        self.blink_flag = True
                        now = time.time()
                        # Double blink = right click
                        if now - self.last_blink_time < self.right_click_timeout:
                            pyautogui.click(button='right')
                            self.status.emit("Right Click (double blink)")
                            self.last_blink_time = 0
                        else:
                            pyautogui.click()
                            self.status.emit("Left Click (blink)")
                            self.last_blink_time = now
                else:
                    self.blink_flag = False

                # Track gaze & move mouse if tracking ON
                if self.tracking:
                    gaze = self.relative_eye_position(lm)
                    raw_x, raw_y = gaze[0], gaze[1]

                    # Kalman filter smoothing
                    smooth_x = self.kf_x.update(raw_x)
                    smooth_y = self.kf_y.update(raw_y)

                    # EMA smoothing on top
                    if self.ema_x is None:
                        self.ema_x = smooth_x
                        self.ema_y = smooth_y
                    else:
                        self.ema_x = SMOOTHING_ALPHA * smooth_x + (1 - SMOOTHING_ALPHA) * self.ema_x
                        self.ema_y = SMOOTHING_ALPHA * smooth_y + (1 - SMOOTHING_ALPHA) * self.ema_y

                    screen_xy = self.map_to_screen(np.array([self.ema_x, self.ema_y]))

                    # Draw gaze point on preview frame
                    preview_frame = frame.copy()
                    preview_x = int(screen_xy[0] * frame.shape[1] / self.screen_w)
                    preview_y = int(screen_xy[1] * frame.shape[0] / self.screen_h)
                    cv2.circle(preview_frame, (preview_x, preview_y), 15, (0, 0, 255), 3)
                    cv2.putText(preview_frame, "Gaze Point", (preview_x + 10, preview_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Preview", preview_frame)

                    pyautogui.moveTo(int(screen_xy[0]), int(screen_xy[1]), duration=0)
                else:
                    # Show camera feed without gaze point when not tracking
                    cv2.imshow("Preview", frame)
            else:
                # No face landmarks - just show camera feed
                cv2.imshow("Preview", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit camera preview
                self.running = False
                break

        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.status.emit("Camera stopped.")

    def stop_thread(self):
        self.running = False

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.tracker = EyeTrackerThread()
        self.tracker.start()

        self.label = QLabel("Status: Idle")
        btn_calib = QPushButton("Calibrate")
        btn_track = QPushButton("Start / Stop Tracking")

        btn_calib.clicked.connect(self.tracker.calibrate)
        btn_track.clicked.connect(self.toggle_tracking)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(btn_calib)
        layout.addWidget(btn_track)
        self.setLayout(layout)

        self.tracker.status.connect(lambda m: self.label.setText("Status: " + m))

        self.setWindowTitle("Eye Control UI")
        self.setFixedSize(320, 160)

    def toggle_tracking(self):
        self.tracker.tracking = not self.tracker.tracking
        self.label.setText("Status: Tracking ON" if self.tracker.tracking else "Status: Tracking OFF")

    def closeEvent(self, event):
        self.tracker.stop_thread()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = App()
    ui.show()
    sys.exit(app.exec())
