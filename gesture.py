import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Program modes
MODES = {
    "GESTURE": "Gesture Control Mode",
    "DRAWING": "Drawing Mode", 
    "MOUSE": "Mouse Control Mode"
}
current_mode = "GESTURE"

# Gesture state
last_gesture = None
gesture_cooldown = 0

# Drawing state
drawing_canvas = None
drawing_color = (0, 255, 0)  # Green
drawing_thickness = 3
start_point = None
drawing_type = None  # 'box', 'circle', 'freehand'

# Mouse control state
mouse_smoothing = []
SMOOTHING_BUFFER_SIZE = 5
mouse_click_cooldown = 0

def get_finger_status(hand_landmarks):
    """Returns list of which fingers are up [thumb, index, middle, ring, pinky]"""
    fingers = []
    
    # Thumb (check x-coordinate instead of y)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    tip_ids = [8, 12, 16, 20]
    for tip_id in tip_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def detect_gesture(hand_landmarks):
    """Detect specific gestures"""
    fingers = get_finger_status(hand_landmarks)
    
    # Peace sign (index + middle up)
    if fingers == [0, 1, 1, 0, 0]:
        return "peace"
    
    # Thumbs up
    elif fingers == [1, 0, 0, 0, 0]:
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        if thumb_tip.y < thumb_base.y:
            return "thumbs_up"
        else:
            return "thumbs_down"
    
    # Fist (no fingers up)
    elif fingers == [0, 0, 0, 0, 0]:
        return "fist"
    
    # All fingers up
    elif fingers == [1, 1, 1, 1, 1]:
        return "open_palm"
    
    # Pointing finger only
    elif fingers == [0, 1, 0, 0, 0]:
        return "pointing"
    
    # Three fingers up (thumb, index, middle)
    elif fingers == [1, 1, 1, 0, 0]:
        return "three_fingers"
    
    # Two fingers (index and thumb) - pinch
    elif fingers == [1, 1, 0, 0, 0]:
        return "pinch"
    
    return None

def get_finger_position(hand_landmarks, frame_shape, finger_tip=8):
    """Get finger tip position in screen coordinates"""
    h, w, _ = frame_shape
    finger_tip = hand_landmarks.landmark[finger_tip]
    x = int(finger_tip.x * w)
    y = int(finger_tip.y * h)
    return (x, y)

def handle_gesture_mode(gesture):
    """Handle actions in gesture control mode"""
    if gesture == "peace":
        print("✌️ PEACE SIGN - Opening Settings...")
        pyautogui.hotkey('win', 'i')
    
    elif gesture == "thumbs_up":
        print("👍 THUMBS UP - Volume Up")
        pyautogui.press('volumeup')
    
    elif gesture == "thumbs_down":
        print("👎 THUMBS DOWN - Volume Down")
        pyautogui.press('volumedown')
    
    elif gesture == "fist":
        print("✊ FIST - Left Click")
        pyautogui.click()
    
    elif gesture == "open_palm":
        print("✋ OPEN PALM - Screenshot")
        pyautogui.hotkey('win', 'shift', 's')
    
    elif gesture == "three_fingers":
        print("🤟 THREE FINGERS - Switch Tabs")
        pyautogui.hotkey('alt', 'tab')

def handle_drawing_mode(hand_landmarks, frame, gesture):
    """Handle drawing actions"""
    global drawing_canvas, start_point, drawing_type
    
    # Initialize canvas if needed
    if drawing_canvas is None:
        drawing_canvas = np.zeros_like(frame)
    
    index_tip = get_finger_position(hand_landmarks, frame.shape)
    
    if gesture == "pointing" and not start_point:
        # Start freehand drawing
        start_point = index_tip
        drawing_type = 'freehand'
        print("🎨 Started Freehand Drawing")
    
    elif gesture == "peace" and not start_point:
        # Start box drawing
        start_point = index_tip
        drawing_type = 'box'
        current_drawing = drawing_canvas.copy()
        print("📦 Started Box Drawing")
    
    elif gesture == "thumbs_up" and not start_point:
        # Start circle drawing
        start_point = index_tip
        drawing_type = 'circle'
        current_drawing = drawing_canvas.copy()
        print("⭕ Started Circle Drawing")
    
    elif gesture == "fist" and start_point:
        # Finalize current drawing
        if drawing_type == 'box' and start_point:
            cv2.rectangle(drawing_canvas, start_point, index_tip, drawing_color, drawing_thickness)
        elif drawing_type == 'circle' and start_point:
            radius = int(math.sqrt((index_tip[0] - start_point[0])**2 + (index_tip[1] - start_point[1])**2))
            cv2.circle(drawing_canvas, start_point, radius, drawing_color, drawing_thickness)
        
        start_point = None
        drawing_type = None
        print("💾 Drawing Saved")
    
    elif gesture == "three_fingers":
        # Clear drawing
        drawing_canvas = np.zeros_like(frame)
        start_point = None
        drawing_type = None
        print("🗑️ Drawing Cleared")
    
    # Handle real-time drawing
    if start_point:
        if drawing_type == 'freehand':
            cv2.line(drawing_canvas, start_point, index_tip, drawing_color, drawing_thickness)
            start_point = index_tip
        
        elif drawing_type == 'box':
            temp_canvas = drawing_canvas.copy()
            cv2.rectangle(temp_canvas, start_point, index_tip, drawing_color, drawing_thickness)
            frame[:] = cv2.addWeighted(frame, 0.7, temp_canvas, 0.3, 0)
        
        elif drawing_type == 'circle':
            temp_canvas = drawing_canvas.copy()
            radius = int(math.sqrt((index_tip[0] - start_point[0])**2 + (index_tip[1] - start_point[1])**2))
            cv2.circle(temp_canvas, start_point, radius, drawing_color, drawing_thickness)
            frame[:] = cv2.addWeighted(frame, 0.7, temp_canvas, 0.3, 0)

def handle_mouse_mode(hand_landmarks, frame, gesture):
    """Handle mouse pointer control"""
    global mouse_smoothing, mouse_click_cooldown
    
    index_tip = get_finger_position(hand_landmarks, frame.shape)
    
    # Convert webcam coordinates to screen coordinates
    h, w, _ = frame.shape
    screen_x = np.interp(index_tip[0], [0, w], [0, screen_width])
    screen_y = np.interp(index_tip[1], [0, h], [0, screen_height])
    
    # Smooth mouse movement
    mouse_smoothing.append((screen_x, screen_y))
    if len(mouse_smoothing) > SMOOTHING_BUFFER_SIZE:
        mouse_smoothing.pop(0)
    
    # Average the positions for smooth movement
    avg_x = int(np.mean([pos[0] for pos in mouse_smoothing]))
    avg_y = int(np.mean([pos[1] for pos in mouse_smoothing]))
    
    # Move mouse pointer
    pyautogui.moveTo(avg_x, avg_y, duration=0.1)
    
    # Handle clicks
    mouse_click_cooldown = max(0, mouse_click_cooldown - 1)
    
    if gesture == "fist" and mouse_click_cooldown == 0:
        print("🖱️ Left Click")
        pyautogui.click()
        mouse_click_cooldown = 20
    
    elif gesture == "peace" and mouse_click_cooldown == 0:
        print("🖱️ Right Click")
        pyautogui.rightClick()
        mouse_click_cooldown = 20
    
    elif gesture == "pinch" and mouse_click_cooldown == 0:
        print("🖱️ Double Click")
        pyautogui.doubleClick()
        mouse_click_cooldown = 20
    
    # Draw cursor on frame
    cv2.circle(frame, index_tip, 10, (255, 0, 0), -1)
    cv2.circle(frame, index_tip, 8, (0, 0, 255), 2)

def switch_mode(direction=1):
    """Switch between modes"""
    global current_mode, drawing_canvas, start_point, mouse_smoothing
    
    modes_list = list(MODES.keys())
    current_index = modes_list.index(current_mode)
    new_index = (current_index + direction) % len(modes_list)
    current_mode = modes_list[new_index]
    
    # Reset state when switching modes
    start_point = None
    mouse_smoothing = []
    
    print(f"\n🔄 Switched to: {MODES[current_mode]}")
    return current_mode

print("🎮 Advanced Hand Gesture Control Started!")
print("=" * 50)
print("MODE CONTROLS:")
print("M - Next Mode | N - Previous Mode")
print("1 - Gesture Mode | 2 - Drawing Mode | 3 - Mouse Mode")
print("\nCURRENT MODES:")
for mode_key, mode_desc in MODES.items():
    print(f"  {mode_key} - {mode_desc}")
print("\n" + "=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    gesture_cooldown = max(0, gesture_cooldown - 1)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand skeleton
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            # Detect gesture
            gesture = detect_gesture(hand_landmarks)
            
            # Handle different modes
            if current_mode == "GESTURE":
                if gesture and gesture != last_gesture and gesture_cooldown == 0:
                    handle_gesture_mode(gesture)
                    last_gesture = gesture
                    gesture_cooldown = 20
            
            elif current_mode == "DRAWING":
                handle_drawing_mode(hand_landmarks, frame, gesture)
            
            elif current_mode == "MOUSE":
                handle_mouse_mode(hand_landmarks, frame, gesture)
            
            # Display current gesture and mode
            status_text = f"Mode: {MODES[current_mode]}"
            if gesture:
                status_text += f" | Gesture: {gesture.upper()}"
            
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display mode-specific instructions
            instructions_y = 60
            if current_mode == "GESTURE":
                cv2.putText(frame, "Gestures: Peace=Settings, Thumbs=Volume, Fist=Click", 
                           (10, instructions_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif current_mode == "DRAWING":
                cv2.putText(frame, "Point=Draw, Peace=Box, Thumbs=Circle, Fist=Save, 3Fingers=Clear", 
                           (10, instructions_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif current_mode == "MOUSE":
                cv2.putText(frame, "Move=Pointer, Fist=LeftClick, Peace=RightClick, Pinch=DoubleClick", 
                           (10, instructions_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        last_gesture = None
        if current_mode == "DRAWING":
            start_point = None
    
    # Apply drawing canvas to frame in drawing mode
    if current_mode == "DRAWING" and drawing_canvas is not None:
        frame = cv2.add(frame, drawing_canvas)
    
    # Display mode in corner
    cv2.putText(frame, f"MODE: {current_mode}", (frame.shape[1] - 200, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    # Bottom instructions
    cv2.putText(frame, "Q=Quit | M=Next Mode | N=Prev Mode | 1/2/3=Quick Switch", 
               (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imshow('Advanced Hand Gesture Control', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):  # Next mode
        switch_mode(1)
    elif key == ord('n'):  # Previous mode
        switch_mode(-1)
    elif key == ord('1'):  # Quick switch to Gesture mode
        if current_mode != "GESTURE":
            current_mode = "GESTURE"
            print(f"\n🔄 Switched to: {MODES[current_mode]}")
    elif key == ord('2'):  # Quick switch to Drawing mode
        if current_mode != "DRAWING":
            current_mode = "DRAWING"
            print(f"\n🔄 Switched to: {MODES[current_mode]}")
    elif key == ord('3'):  # Quick switch to Mouse mode
        if current_mode != "MOUSE":
            current_mode = "MOUSE"
            print(f"\n🔄 Switched to: {MODES[current_mode]}")

cap.release()
cv2.destroyAllWindows()