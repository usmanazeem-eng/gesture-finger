try:
    from mediapipe.python.solutions import hands as mp_hands
    print("Success: Imported using mediapipe.python.solutions")
except Exception as e:
    print(f"Failed mediapipe.python.solutions: {e}")

try:
    import mediapipe as mp
    print(f"MediaPipe attributes: {dir(mp)}")
except Exception as e:
    print(f"Failed check dir: {e}")
