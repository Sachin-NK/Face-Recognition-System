# Face Recognition System
This project implements a real-time face recognition system using a pre-trained DeepFace model, OpenCV, and a custom database for face encodings. It supports recognizing faces from a live webcam feed and identifying them based on stored encodings.

## Features

- Real-time Face Detection: Uses OpenCV's Haar Cascade Classifier for face detection.
- Face Recognition: Leverages DeepFace with the Facenet model for generating embeddings and matching faces.
- Custom Database: Allows storing and loading pre-encoded face data from a .pkl file.
- Threshold-Based Matching: Configurable similarity threshold to determine face matches.

