import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from yolov5 import detect

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Streamlit app
st.title("Object Detection using YOLOv5")
st.write("Upload an image and the app will identify objects in it.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting...")

    # Convert image to a format compatible with YOLOv5
    image = np.array(image)

    # Run YOLOv5 detection
    results = model(image)

    # Render results
    st.write("Detected Objects:")
    results.print()  # Print results to console
    results.render()  # Draw boxes on the image

    # Convert the result image with boxes to a format displayable by Streamlit
    result_img = Image.fromarray(results.ims[0])
    st.image(result_img, caption="Detected Objects", use_column_width=True)
