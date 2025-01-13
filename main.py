import cv2
import pickle
import os
import numpy as np
from deepface import DeepFace

ENCODINGS_FILE = "encodings.pkl"

# Load the saved face encodings and names.
def load_encodings(encodings_file):
    
    if not encodings_file or not os.path.exists(encodings_file):
        raise FileNotFoundError(f"Encoding file '{encodings_file}' not found.")
    with open(encodings_file, 'rb') as f:
        return pickle.load(f)

# Real-time face recognition
def recognize_faces(encodings_file):
    
    known_face_encodings, known_face_names = load_encodings(encodings_file)

    print("Starting webcam. Press 'q' to quit.")
    video_capture = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = frame[y:y+h, x:x+w]
            try:
                # Generate embedding for the detected face
                embedding = DeepFace.represent(face_roi, model_name="Facenet")[0]["embedding"]
                embedding = np.array(embedding)

                # Find the best match for the embedding
                distances = [np.linalg.norm(encoding - embedding) for encoding in known_face_encodings]
                best_match_index = np.argmin(distances)

                # Match if within threshold
                if distances[best_match_index] < 0.6:  
                    name = known_face_names[best_match_index]
                    color = (0, 255, 0)  # Green for recognized faces
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown faces

            except Exception as e:
                # Handle any errors during embedding generation or matching
                name = "Error"
                color = (0, 0, 255)
                print(f"Error recognizing face: {e}")

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Run the face recognition
recognize_faces(ENCODINGS_FILE)
