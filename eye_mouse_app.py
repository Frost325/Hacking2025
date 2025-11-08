import sys
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import keyboard   # For Q toggle
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt6.QtCore import QThread, pyqtSignal

SMOOTHING_ALPHA = 0.25

class EyeTrackerThread(QThread):
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.tracking = False
        self.calibrated = False
        self.T = None
        self.prev_smoothed = None
        self.blink_flag = False

        self.screen_w, self.screen_h = pyautogui.size()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = None

    def relative_eye_position(self, lm):
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]

        ix = np.mean([lm[i].x for i in LEFT_IRIS + RIGHT_IRIS])
        iy = np.mean([lm[i].y for i in LEFT_IRIS + RIGHT_IRIS])

        eye_left = lm[33].x
        eye_right = lm[133].x
        eye_top = lm[159].y
        eye_bottom = lm[145].y

        nx = (ix - eye_left) / (eye_right - eye_left)
        ny = (iy - eye_top) / (eye_bottom - eye_top)

        return np.array([nx, ny])

    def map_to_screen(self, norm_xy):
        v = np.array([norm_xy[0], norm_xy[1], 1.0])
        scr = v @ self.T
        scr[0] = np.clip(scr[0], 0, self.screen_w-1)
        scr[1] = np.clip(scr[1], 0, self.screen_h-1)
        return scr.astype(int)

    def eye_aspect_ratio(self, lm, left=True):
        if left:
            top = lm[159]; bottom = lm[145]
            left_pt = lm[33]; right_pt = lm[133]
        else:
            top = lm[386]; bottom = lm[374]
            left_pt = lm[362]; right_pt = lm[263]

        v = np.linalg.norm([top.x-bottom.x, top.y-bottom.y])
        h = np.linalg.norm([left_pt.x-right_pt.x, left_pt.y-right_pt.y])
        return v / h

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

        for name, (nx, ny) in CAP_POINTS:
            self.status.emit(f"Look at {name}, press SPACE")
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                h, w = frame.shape[:2]

                # >>>>>>> ADD HEAD POSITION BOX <<<<<<<<
                box_w = int(w * 0.45)
                box_h = int(h * 0.45)
                x1 = w//2 - box_w//2
                y1 = h//2 - box_h//2
                x2 = w//2 + box_w//2
                y2 = h//2 + box_h//2
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "Keep your head centered in box",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                # Dot to look at
                cv2.circle(frame, (int(nx*w), int(ny*h)), 18, (0,255,0), 2)

                cv2.imshow("Calibration", frame)

                k = cv2.waitKey(1) & 0xFF
                if k == ord(' '):
                    res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if res.multi_face_landmarks:
                        lm = res.multi_face_landmarks[0].landmark
                        cam_pts.append(self.relative_eye_position(lm))
                        scr_pts.append(np.array([nx*self.screen_w, ny*self.screen_h]))
                        break
                elif k == ord('q'):
                    cv2.destroyWindow("Calibration")
                    return

        cv2.destroyWindow("Calibration")

        src = np.array(cam_pts)
        dst = np.array(scr_pts)
        A = np.hstack([src, np.ones((len(src),1))])
        self.T, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)

        self.calibrated = True
        self.status.emit("Calibration complete.")

    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.status.emit("Camera failed.")
            return
        self.status.emit("Camera on.")

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

                # Blink → click
                ear = (self.eye_aspect_ratio(lm, True) + self.eye_aspect_ratio(lm, False)) / 2
                if ear < 0.21:
                    if not self.blink_flag:
                        self.blink_flag = True
                        pyautogui.click()
                else:
                    self.blink_flag = False

                if self.tracking:
                    gaze = self.relative_eye_position(lm)
                    screen_xy = self.map_to_screen(gaze)

                    if self.prev_smoothed is None:
                        self.prev_smoothed = screen_xy.astype(float)
                    else:
                        self.prev_smoothed = SMOOTHING_ALPHA*screen_xy + (1-SMOOTHING_ALPHA)*self.prev_smoothed

                    pyautogui.moveTo(int(self.prev_smoothed[0]), int(self.prev_smoothed[1]), duration=0)

        if self.cap.isOpened():
            self.cap.release()
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
        self.setFixedSize(300, 150)

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
