import streamlit as st
import numpy as np
from PIL import Image
import time
from keras.models import load_model
from ultralytics import YOLO
import cv2
import tempfile
import os

# -------------------------------
# Load trained CNN model
# -------------------------------
@st.cache_resource
def load_cnn_model():
    with st.spinner("Loading CNN model..."):
        time.sleep(1)
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(BASE_DIR, "best_model.keras")
            model = load_model(model_path)
        except Exception as e:
            st.error(f"CNN Model file not found or failed to load: {e}")
            model = None
    return model


# -------------------------------
# Load trained YOLO model
# -------------------------------
@st.cache_resource
def load_yolo_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "yolo11n.pt")  # ✅ Fixed: removed load_model() call
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"YOLO Model file not found or failed to load: {e}")
        model = None
    return model


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Aerial Detection App", layout="wide")
st.title("Aerial Object Detection – Bird vs Drone")

left_col, right_col = st.columns(2, gap="large")


# --------------------------------------------------
# LEFT COLUMN – CNN Model
# --------------------------------------------------
with left_col:
    cnn_model = load_cnn_model()
    st.header("CNN / MobileNet Model")
    st.markdown('<p>Upload an image for Bird vs Drone classification</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Image (CNN)", type=["jpg", "png", "jpeg"])

    if uploaded is not None:
        st.info("Processing uploaded image...")

        # Progress bar
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        img = Image.open(uploaded).convert("RGB").resize((224, 224))
        st.image(img, caption="Uploaded Image", use_container_width=True)

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        if cnn_model is not None:
            st.write("Running prediction...")

            with st.spinner("Analyzing..."):
                time.sleep(0.5)
                pred = cnn_model.predict(arr)[0][0]

            # Confidence calculation
            bird_conf = float(pred) * 100
            drone_conf = float(100 - bird_conf)

            # Determine class
            if pred > 0.5:
                label = "Bird"
                conf = bird_conf
            else:
                label = "Drone"
                conf = drone_conf

            # Display result
            st.success(f"**Prediction:** {label}")

            st.write(f"### Confidence: **{conf:.2f}%**")
            st.progress(int(conf))

            st.write("### Probability Breakdown")
            st.write(f"**Bird:** {bird_conf:.2f}%")
            st.write(f"**Drone:** {drone_conf:.2f}%")
        else:
            st.error("CNN model could not be loaded. Please check the model file.")


# --------------------------------------------------
# RIGHT COLUMN – YOLO Model
# --------------------------------------------------
with right_col:
    yolo_model = load_yolo_model()
    st.header("YOLO Detection Model")
    st.write("Upload an image or video to detect birds and drones.")

    option = st.selectbox(
        "Choose Input Type:",
        ["Image", "Video", "Webcam"]
    )

    # --------------------------------------------------
    # IMAGE DETECTION
    # --------------------------------------------------
    if option == "Image":
        uploaded_file = st.file_uploader("Upload an Image (YOLO)", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Detect"):
                if yolo_model is not None:
                    results = yolo_model.predict(image)
                    result_image = results[0].plot()
                    st.image(result_image, caption="Detection Result", use_container_width=True)
                else:
                    st.error("YOLO model could not be loaded. Please check the model file.")

    # --------------------------------------------------
    # VIDEO DETECTION
    # --------------------------------------------------
    elif option == "Video":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()

            if st.button("Detect"):
                if yolo_model is not None:
                    stframe = st.empty()
                    cap = cv2.VideoCapture(tfile.name)

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = yolo_model.predict(frame)
                        result_frame = results[0].plot()
                        stframe.image(result_frame, channels="BGR", use_container_width=True)

                    cap.release()
                    st.success("Video processing complete!")
                else:
                    st.error("YOLO model could not be loaded. Please check the model file.")

    # --------------------------------------------------
    # WEBCAM DETECTION
    # --------------------------------------------------
    elif option == "Webcam":
        st.warning("Webcam is not supported on Streamlit Cloud. Please run locally.")
        if st.button("Start Webcam (Local Only)"):
            if yolo_model is not None:
                stframe = st.empty()
                cap = cv2.VideoCapture(0)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Could not access webcam.")
                        break

                    results = yolo_model.predict(frame)
                    frame = results[0].plot()
                    stframe.image(frame, channels="BGR", use_container_width=True)

                cap.release()
            else:
                st.error("YOLO model could not be loaded. Please check the model file.")
