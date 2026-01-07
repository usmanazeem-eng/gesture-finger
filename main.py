import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np 
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
# SETTINGS 
SCROLL_SPEED = 50
SCROLL_THRESHOLD = 0.05
SMOOTHING = 4     
MARGIN = 0.15     
ZOOM_THRESHOLD = 0.1
CLICK_THRESHOLD = 0.05

pyautogui.PAUSE = 0    
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# LOAD MODEL
try:
    with open("gesture_model.pkl", "rb") as f:
        gesture_model = pickle.load(f)
    print("Model loaded successfully!")
except:
    gesture_model = None
    print("Warning: gesture_model.pkl not found! Falling back to distance-based rules.")

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
prev_thumb_y = None
scroll_enabled = True
last_action_time = 0

# Feedback Display
feedback_text = ""
feedback_expiry = 0

def cooldown(sec=0.4):
    global last_action_time
    if time.time() - last_action_time > sec:
        last_action_time = time.time()
        return True
    return False

print("--- Advanced Gesture System Started ---")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            # ========== MOUSE MOVE (Index) ==========
            # Added MARGIN scaling to reach corners easily
            hx, hy = lm[8].x, lm[8].y
            ix = np.clip((hx - MARGIN) / (1 - 2 * MARGIN), 0, 1)
            iy = np.clip((hy - MARGIN) / (1 - 2 * MARGIN), 0, 1)

            cur_x = prev_x + (ix - prev_x) / SMOOTHING
            cur_y = prev_y + (iy - prev_y) / SMOOTHING
            prev_x, prev_y = cur_x, cur_y
            pyautogui.moveTo(int(cur_x * screen_w), int(cur_y * screen_h))

            # ========== SCROLL (Thumb UP / DOWN) ==========
            thumb_y = lm[4].y
            if scroll_enabled and prev_thumb_y is not None:
                diff = prev_thumb_y - thumb_y
                if diff > SCROLL_THRESHOLD:
                    pyautogui.scroll(SCROLL_SPEED)
                elif diff < -SCROLL_THRESHOLD:
                    pyautogui.scroll(-SCROLL_SPEED)
            prev_thumb_y = thumb_y


            # ========== GESTURE CLASSIFICATION (AI MODEL) ==========
            current_gesture = None
            if gesture_model:
                try:
                    # Collect 2D landmarks for prediction (X, Y only)
                    features = []
                    for landmark in lm:
                        features.extend([landmark.x, landmark.y])
                    
                    # Predict
                    prediction = gesture_model.predict([features])[0]
                    current_gesture = str(prediction).lower()
                    
                    # Debug: Show AI prediction on screen
                    cv2.putText(frame, f"AI Sees: {current_gesture.upper()}", (10, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                except Exception as e:
                    print(f"Prediction Error: {e}")

            # ========== ACTION LOGIC (AI MODEL BASED) ==========
            if current_gesture and current_gesture != "idle" and cooldown(0.5):
                if current_gesture == "jump":
                    pyautogui.press('up')
                    feedback_text = "JUMP"
                    feedback_expiry = time.time() + 1.0
                elif current_gesture == "down":
                    pyautogui.press('down')
                    feedback_text = "SLIDE"
                    feedback_expiry = time.time() + 1.0
                elif current_gesture == "left":
                    pyautogui.press('left')
                    feedback_text = "LEFT"
                    feedback_expiry = time.time() + 1.0
                elif current_gesture == "right":
                    pyautogui.press('right')
                    feedback_text = "RIGHT"
                    feedback_expiry = time.time() + 1.0

            # Show Persistent Feedback
            if time.time() < feedback_expiry:
                cv2.putText(frame, feedback_text, (w//2 - 60, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            # ========== ACTION LOGIC ==========
            # Left Click: Very strict (Thumb + Index touching)
            d_click = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
            
            # Using a very strict threshold (0.04) to prevent auto-clicks
            # Model is used only as a secondary confirmation
            is_click_gesture = (d_click < 0.04) 
            if gesture_model and is_click_gesture:
                # If model is available, it should ideally agree it's a click
                # But we don't block it entirely because model might be biased
                pass

            if is_click_gesture and cooldown():
                pyautogui.click()
                cv2.putText(frame, "LEFT CLICK", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Right Click: Index + Middle touching
            d_right = math.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y)
            
            # Strict threshold for right click as well
            is_right_gesture = (d_right < 0.04) or (current_gesture == "peace" and d_right < 0.06)
            
            # Ensure they don't overlap: If Thumb-Index is closer, prioritize Left Click
            if d_click < d_right:
                is_right_gesture = False

            if is_right_gesture and cooldown():
                pyautogui.rightClick()
                cv2.putText(frame, "RIGHT CLICK", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # ========== ZOOM (Specific Gesture: Thumb + Index) ==========
            # Using thumb and index distance for more deliberate zoom
            d_zoom = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
            # Only zoom if other fingers are closed to avoid accidental trigger
            middle_closed = lm[12].y > lm[10].y
            if middle_closed:
                if d_zoom > 0.15 and cooldown(0.5):
                    pyautogui.hotkey('ctrl', '+')
                    cv2.putText(frame, "ZOOM IN", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,0), 2)
                elif d_zoom < 0.04 and cooldown(0.5):
                    pyautogui.hotkey('ctrl', '-')
                    cv2.putText(frame, "ZOOM OUT", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,0), 2)

            # ========== PALM / FIST ==========
            fingers_up = lm[8].y < lm[6].y and lm[12].y < lm[10].y
            if fingers_up:
                scroll_enabled = True
            else:
                scroll_enabled = False

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AI Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
