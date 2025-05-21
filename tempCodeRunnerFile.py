import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Allow up to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(0)

# Mapping function for gesture to alphabet
def recognize_finger_alphabet(hand_landmarks):
    # Helper function to calculate distances between landmarks
    def dist(p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # Basic gesture recognition logic (can be extended)
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Extended gesture recognition logic
    if dist(thumb_tip, index_finger_tip) < 0.1 and dist(middle_finger_tip, ring_finger_tip) > 0.2:
        return "D"
    elif dist(index_finger_tip, middle_finger_tip) < 0.1 and dist(ring_finger_tip, pinky_tip) > 0.2:
        return "V"
    elif dist(thumb_tip, pinky_tip) < 0.1:
        return "O"
    return "?"  # Default value when no gesture is recognized

# Main Loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks: # Iterate through each hand
            # Recognize the finger alphabet
            letter = recognize_finger_alphabet(hand_landmarks)

            # Display recognised letter on the screen
            cv2.putText(image, letter, (50, 50),  # Display letter at a fixed position
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting image
    cv2.imshow('Hand Recognition', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()