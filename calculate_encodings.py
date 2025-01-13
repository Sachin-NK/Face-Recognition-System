import os
import cv2
import pickle
import numpy as np

IMAGES_FOLDER = r"path_to_img_folder" 
ENCODINGS_FILE = "encodings.pkl"  

def extract_face_encoding(face_image):    
    face_resized = cv2.resize(face_image, (96, 96)) 
    face_flattened = face_resized.flatten()  # Flatten the image into a 1D array
    return face_flattened

def calculate_and_save_encodings(images_folder, encodings_file):
    known_face_encodings = []
    known_face_names = []

    # Initialize OpenCV's Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Traverse all subfolders
    for person_name in os.listdir(images_folder):
        person_folder = os.path.join(images_folder, person_name)
        if os.path.isdir(person_folder): 
            print(f"Processing folder: {person_name}")
            for filename in os.listdir(person_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(person_folder, filename)
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"Warning: Could not read image {image_path}")
                            continue
                        
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for detection
                        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                        for (x, y, w, h) in faces:
                            face_image = image[y:y+h, x:x+w]  # Crop the face from the image
                            face_encoding = extract_face_encoding(face_image)
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(person_name)  # Use subfolder name as the label
                    
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")

    # Save encodings and names to a file
    try:
        with open(encodings_file, 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        print(f"Encodings saved to {encodings_file}")
    except Exception as e:
        print(f"Error saving encodings to file: {e}")

calculate_and_save_encodings(IMAGES_FOLDER, ENCODINGS_FILE)
