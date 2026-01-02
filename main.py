import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logs
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np # Import numpy to avoid secondary issues

# ================= SETTINGS =================
SCROLL_SPEED = 50
SCROLL_THRESHOLD = 0.05
SMOOTHING = 6
ZOOM_THRESHOLD = 0.05
CLICK_THRESHOLD = 0.03
# ===========================================

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
prev_thumb_y = None
scroll_enabled = True
last_action_time = 0

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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            # ========== MOUSE MOVE (Index) ==========
            ix, iy = lm[8].x, lm[8].y
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

            # ========== LEFT CLICK ==========
            d_click = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
            if d_click < CLICK_THRESHOLD and cooldown():
                pyautogui.click()
                cv2.putText(frame, "LEFT CLICK", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # ========== RIGHT CLICK ==========
            d_right = math.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y)
            if d_right < CLICK_THRESHOLD and cooldown():
                pyautogui.rightClick()
                cv2.putText(frame, "RIGHT CLICK", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # ========== ZOOM ==========
            d_zoom = math.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y)
            if d_zoom > ZOOM_THRESHOLD and cooldown(0.5):
                pyautogui.hotkey('ctrl', '+')
            elif d_zoom < 0.02 and cooldown(0.5):
                pyautogui.hotkey('ctrl', '-')

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
