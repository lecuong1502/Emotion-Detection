import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# For webcam feed   
cap = cv2.VideoCapture(0)  # Open the default camera

#Smile detection parameters
SMILE_CONFIDENCE_FRAMES = 5  # Number of consecutive frames required
smile_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3, 
        minNeighbors=5, 
        minSize=(100, 100)  # Larger face detection area
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        roi_gray = gray[y+int(h/2):y+h, x:x+w]  # Focus on lower face
        roi_color = frame[y+int(h/2):y+h, x:x+w]
        
        # Detect smile with strict parameters
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=35,  # Increased from 20
            minSize=(50, 50),  # Larger minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        #Smile validation logic
        if len(smiles) > 0:
            smile_counter = min(smile_counter + 1, SMILE_CONFIDENCE_FRAMES)
        else:
            smile_counter = max(smile_counter - 1, 0)
            
        # Display results
        if smile_counter >= SMILE_CONFIDENCE_FRAMES:
            cv2.putText(frame, 'Confirmed Smile', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif len(smiles) > 0:
            cv2.putText(frame, 'Possible Smile', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            
        # Draw smile rectangles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    #Display the resulting frame
    cv2.imshow('Advanced Smile Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#Release the capture
cap.release()
cv2.destroyAllWindows()  # Close all OpenCV windows