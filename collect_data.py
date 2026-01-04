import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import csv

# ================= SETTINGS =================
GESTURE_NAME = "click" # ISKO CHANGE KAREIN: "click", "scroll", "idle", etc.
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
count = 0

# Pehle se kitne samples hain check karein
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == GESTURE_NAME:
                count += 1

print(f"--- Data Collector Started ---")
print(f"Gesture Name: {GESTURE_NAME}")
print(f"Pehle ke samples: {count}")
print("\nInstructions:")
print("1. Camera ke samne hath dikhayein.")
print("2. 'S' press karein sample save karne ke liye.")
print("3. Har gesture ke liye kam se kam 30-50 samples lein.")
print("4. 'Q' press karein band karne ke liye.")

while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    status_text = "Hand Not Detected"
    status_color = (0, 0, 255)
    
    if results.multi_hand_landmarks:
        status_text = "Hand Detected"
        status_color = (0, 255, 0)
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Key check for saving
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Saare 21 points save karein
                data = [GESTURE_NAME]
                for lm in hand_lms.landmark:
                    data.extend([lm.x, lm.y, lm.z]) # X, Y, Z coordinates
                
                with open(DATA_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                
                count += 1
                print(f"Sample #{count} saved!")

    # UI Overlay
    cv2.rectangle(frame, (0,0), (300, 120), (0,0,0), -1)
    cv2.putText(frame, f"Gesture: {GESTURE_NAME}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Samples: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, "S: Save | Q: Quit", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Hand Data Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
