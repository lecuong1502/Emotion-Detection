import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP,
               mp_hands.HandLandmark.THUMB_TIP]

cap = cv2.VideoCapture(0)  # Open the default camera
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw full hand landmarks
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            for tip_id in FINGER_TIPS:
                landmark = hand_landmarks.landmark[tip_id]
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)  # Green fill
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 2)   # Red border

    cv2.imshow('Finger Highlight', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'Esc' to exit
        break
    
cap.release()
cv2.destroyAllWindows()  # Close all OpenCV windows