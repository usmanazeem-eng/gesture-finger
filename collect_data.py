import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import csv
import os

# Settings
GESTURE_NAME = "peace" # Change this for each gesture you want to train
DATA_FILE = "gestures.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print(f"--- Collecting data for: {GESTURE_NAME} ---")
print("Press 's' to save current hand pose. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Show instructions on screen
            cv2.putText(frame, f"Gesture: {GESTURE_NAME}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Extract 21 landmarks (x, y)
                data = [GESTURE_NAME]
                for lm in hand_lms.landmark:
                    data.extend([lm.x, lm.y])
                
                # Write to CSV
                file_exists = os.path.isfile(DATA_FILE)
                with open(DATA_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                print(f"Saved pose for {GESTURE_NAME}!")

    cv2.imshow("Data Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
