import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)  # Open the default camera
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Analyze the frame for emotion detection
    try:
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,  # Continue even if no face found
            detector_backend='opencv',
            silent=True
        )
        
        if isinstance(analysis, list):
            for idx, face in enumerate(analysis):
                emotion = face['dominant_emotion']
                confidence = face['emotion'][emotion]
                
                text = f"Face {idx+1}: {emotion} ({confidence:.1f}%)"
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                
                cv2.rectangle(
                    frame, 
                    (10, 15 + idx*30), 
                    (10 + text_width, 15 + (idx+1)*30),
                    (255, 255, 255), -1
                )
                
                cv2.putText(
                    frame, text, (10, 30 + idx*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
                )
        else:
            emotion = analysis['dominant_emotion']
            confidence = analysis['emotion'][emotion]
            
            text = f"Emotion: {emotion} ({confidence:.1f}%)"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )
            
            cv2.rectangle(
                frame,
                (10, 15),
                (10 + text_width, 15 + text_height + 10),
                (255, 255, 255), -1
            )  
            
            cv2.putText(
                frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
            )
    except Exception as e:
        print("Detection error:", e)
        cv2.rectangle(frame, (10, 15), (250, 50), (255, 255, 255), -1)
        cv2.putText(frame, "No Face Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows