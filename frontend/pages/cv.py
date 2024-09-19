import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the pre-trained YOLOv8 model (from Ultralytics)
model = YOLO('yolov8n.pt')  # You can replace 'yolov8n.pt' with a different model if needed

st.title("Real-Time YOLO Object Detection with Streamlit")

# Start video stream checkbox
run = st.checkbox('Run', value=False)
FRAME_WINDOW = st.image([])

# Access the webcam (use 0 for the default camera, or a specific index for external cameras)
cap = cv2.VideoCapture(0)  # 0 -> default webcam

# Ensure the stream starts with an initial check
if not cap.isOpened():
    st.error("Webcam not detected. Please check your camera.")
else:
    # Streamlit loop for real-time video feed
    frame_placeholder = st.empty()  # Create a placeholder for video feed

    while run:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to access webcam. Please check your camera settings.")
            break

        # Perform object detection on the frame
        results = model(frame)

        # Check if results were returned correctly
        if results and len(results) > 0:
            # Parse and draw results on the image
            annotated_frame = np.squeeze(results[0].plot())  # Draw boxes and labels

            # Convert from BGR (OpenCV format) to RGB for displaying in Streamlit
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Update the image in the placeholder
            frame_placeholder.image(annotated_frame, channels='RGB')
        else:
            st.error("No objects detected.")

        # # Allow Streamlit to update
        # time.sleep(0.1)  # Adjust the sleep time as needed

    # Release the webcam when done
    cap.release()
