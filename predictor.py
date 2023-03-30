import cv2
import dlib

# Initialize the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Loop to read frames from the video capture object
while True:
    # Read the current frame
    ret, frame = cap.read()
    # If the frame cannot be read, break out of the loop
    if not ret:
        break
    
    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use the face detector to detect faces in the grayscale frame
    faces = detector(gray)

    # Loop over each face detected in the frame
    for face in faces:
        # Use the shape predictor to get the landmarks of the current face
        landmarks = predictor(gray, face)
        
        # Draw lines on the face to show the outline of the face
        for i in range(1, 17):
            x1, y1 = landmarks.part(i-1).x, landmarks.part(i-1).y
            x2, y2 = landmarks.part(i).x, landmarks.part(i).y
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw lines on the face to show the outline of the eyes
        for i in range(37, 43):
            x1, y1 = landmarks.part(i).x, landmarks.part(i).y
            x2, y2 = landmarks.part(i+1).x, landmarks.part(i+1).y
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1 = landmarks.part(36).x, landmarks.part(36).y
        x2, y2 = landmarks.part(41).x, landmarks.part(41).y
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw lines on the face to show the outline of the nose
        for i in range(28, 36):
            x1, y1 = landmarks.part(i).x, landmarks.part(i).y
            x2, y2 = landmarks.part(i+1).x, landmarks.part(i+1).y
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1 = landmarks.part(27).x, landmarks.part(27).y
        x2, y2 = landmarks.part(31).x, landmarks.part(31).y
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw lines on the face to show the outline of the mouth
        for i in range(49, 59):
            x1, y1 = landmarks.part(i).x, landmarks.part(i).y
            x2, y2 = landmarks.part(i+1).x, landmarks.part(i+1).y
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1 = landmarks.part(48).x, landmarks.part(48).y
        x2, y2 = landmarks.part(59).x, landmarks.part(59).y
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Waits for a key event for 1 millisecond, and if the pressed key is "q", the loop is exited
    cv2.imshow("Face Landmarks Detector", frame)
    if cv2.waitKey(1) == ord('q'):
        break
# Releases the video capture object to free up system resources
cap.release()
# Destroys all windows created by the program.
cv2.destroyAllWindows()