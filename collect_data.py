import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import csv

# ================= SETTINGS =================
GESTURES = ["idle", "jump", "down", "left", "right", "click"]
gesture_idx = 0
DATA_FILE = "gestures.csv"
# ============================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print(f"--- Data Collector Started ---")
print("Instructions:")
print("1. Press '1'-'6' to change gesture label.")
print("2. Press 'S' to save sample.")
print("3. Press 'Q' to quit.")

while True:
    GESTURE_NAME = GESTURES[gesture_idx]
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    status_text = "Hand Not Detected"
    status_color = (0, 0, 255)
    
    key = cv2.waitKey(1) & 0xFF
    # Switch Gestures
    if ord('1') <= key <= ord('6'):
        gesture_idx = key - ord('1')
        print(f"Switched to: {GESTURES[gesture_idx]}")

    if results.multi_hand_landmarks:
        status_text = "Hand Detected"
        status_color = (0, 255, 0)
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            if key == ord('s'):
                data = [GESTURE_NAME]
                for lm in hand_lms.landmark:
                    data.extend([lm.x, lm.y])
                
                with open(DATA_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                print(f"Sample saved for {GESTURE_NAME}!")

    # UI Overlay
    cv2.rectangle(frame, (0,0), (400, 130), (0,0,0), -1)
    cv2.putText(frame, f"Current: {GESTURE_NAME}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Keys: 1:idle, 2:jump, 3:down", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"      4:left, 5:right, 6:click", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, status_text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, "S: Save | Q: Quit", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Hand Data Collector", frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
