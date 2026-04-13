import streamlit as st
import numpy as np
from PIL import Image
import time
from keras.models import load_model
from ultralytics import YOLO
import cv2
import tempfile

# print("Current working directory:", os.getcwd())

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_cnn_model():
    with st.spinner("Loading model..."):
        time.sleep(1)
        try:
            model = load_model("models/mobilenet_bird_drone_classifier.keras")
        except FileNotFoundError:
            st.error("Model file not found: models/mobilenet_bird_drone_classifier.keras")
            model = None
    return model


@st.cache_resource
def load_yolo_model():
    try:
        MODEL_PATH = "models/best_yolo_model.pt"  # change if needed
        model = YOLO(MODEL_PATH)
    except FileNotFoundError:
        model = None
    return model

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Aerial Detection App", layout="wide")
st.title("Aerial Object Detection – Bird vs Drone")
left_col, right_col = st.columns(2, gap="large")

# --------------------------------------------------
# LEFT COLUMN – INPUT CONTROLS
# --------------------------------------------------
with left_col:
    model = load_cnn_model()
    st.header("CNN, Mobilenet Model")
    st.markdown('<p class="upload-text">Upload an image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "png"])

    # -----------------------------------------------------
    # When Image Uploaded
    # -----------------------------------------------------
    if uploaded:
        st.info("Processing uploaded image...")

        # Fake progress bar for upload UI
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        img = Image.open(uploaded).resize((224, 224))
        st.image(img, caption="Uploaded Image", use_container_width=True)

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)


        # -----------------------------------------------------
        # Prediction Section
        # -----------------------------------------------------

        st.write("Running prediction...")

        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            pred = model.predict(arr)[0][0]

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

        # -----------------------------------------------------
        # Confidence Score Visualization
        # -----------------------------------------------------
        st.write(f"### Confidence: **{conf:.2f}%**")

        st.progress(int(conf))

        # Side-by-side probabilities
        st.write("### Probability Breakdown")
        st.write(f"**Bird:** {bird_conf:.2f}%")
        st.write(f"**Drone:** {drone_conf:.2f}%")


# --------------------------------------------------
# RIGHT COLUMN – OUTPUT DISPLAY
# --------------------------------------------------
with right_col:
    model = load_yolo_model()
    st.header("YOLO Detection Model")
    st.write("Upload an image or video to detect birds and drones.")
    option = st.selectbox(
        "Choose Input Type:",
        ["Image", "Video", "Webcam"]
    )
    # --------------------------------------------------
    # IMAGE PREDICTION
    # --------------------------------------------------
    if option == "Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Detect"):
                results = model.predict(image)
                result_image = results[0].plot()
                st.image(result_image, caption="Prediction", use_container_width=True)

    # --------------------------------------------------
    # VIDEO PREDICTION
    # --------------------------------------------------
    elif option == "Video":

        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            if st.button("Detect"):
                stframe = st.empty()

                cap = cv2.VideoCapture(tfile.name)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame)
                    result_frame = results[0].plot()

                    stframe.image(result_frame, channels="BGR")

                cap.release()

    # --------------------------------------------------
    # WEBCAM PREDICTION
    # --------------------------------------------------
    elif option == "Webcam":

        if st.button("Start Webcam"):
            stframe = st.empty()

            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame)
                frame = results[0].plot()

                stframe.image(frame, channels="BGR")

            cap.release()




