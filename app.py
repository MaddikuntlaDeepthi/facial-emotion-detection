
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image
from threading import Lock
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="😊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models_v4" / "best_v4.keras"
YUNET_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"

EMOTIONS = [
    "Neutral",
    "Happiness",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
    "Contempt"
]

SCORE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.30
TOP_K = 5000


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_emotion_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Emotion model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    if model.input_shape[1:] != (48, 48, 1):
        raise ValueError(
            f"Unexpected model input shape: {model.input_shape}"
        )

    if model.output_shape[-1] != len(EMOTIONS):
        raise ValueError(
            f"Unexpected model output shape: {model.output_shape}"
        )

    return model


# =========================================================
# CREATE YUNET FACE DETECTOR
# =========================================================

def create_face_detector():

    if not YUNET_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {YUNET_PATH}"
        )

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH),
        "",
        (320, 320),
        SCORE_THRESHOLD,
        NMS_THRESHOLD,
        TOP_K
    )

    return detector


# =========================================================
# EMOTION PREDICTION
# =========================================================

def predict_emotion(face_image, model):

    gray = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )

    normalized = resized.astype("float32") / 255.0

    input_image = np.expand_dims(
        normalized,
        axis=(0, -1)
    )

    predictions = model.predict(
        input_image,
        verbose=0
    )[0]

    emotion_index = int(np.argmax(predictions))

    emotion = EMOTIONS[emotion_index]

    confidence = float(predictions[emotion_index]) * 100

    return emotion, confidence, predictions


# =========================================================
# DRAW FACE BOX AND EMOTION LABEL
# =========================================================

def draw_emotion(frame, face, model):

    x, y, w, h = [int(value) for value in face[:4]]

    height, width = frame.shape[:2]

    x = max(0, x)
    y = max(0, y)

    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x or y2 <= y:
        return frame

    face_crop = frame[y:y2, x:x2]

    if face_crop.size == 0:
        return frame

    emotion, confidence, _ = predict_emotion(
        face_crop,
        model
    )

    label = f"{emotion}: {confidence:.1f}%"

    # Draw face rectangle
    cv2.rectangle(
        frame,
        (x, y),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Calculate label background position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness
    )

    label_y = max(y - 10, text_height + 10)

    # Label background
    cv2.rectangle(
        frame,
        (x, label_y - text_height - 10),
        (x + text_width + 10, label_y + baseline),
        (0, 255, 0),
        -1
    )

    # Emotion text
    cv2.putText(
        frame,
        label,
        (x + 5, label_y - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    return frame


# =========================================================
# PROCESS FRAME
# =========================================================

def process_frame(frame, model, detector, model_lock):

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    detected_count = 0

    if faces is not None:

        for face in faces:

            frame = draw_emotion(
                frame,
                face,
                model
            )

            detected_count += 1

    return frame, detected_count


# =========================================================
# WEBCAM VIDEO PROCESSOR
# =========================================================

class EmotionVideoProcessor(VideoProcessorBase):

    def __init__(self, model):

        self.model = model
        self.detector = create_face_detector()
        self.model_lock = Lock()

    def recv(self, video_frame):

        frame = video_frame.to_ndarray(
            format="bgr24"
        )

        try:

            with self.model_lock:

                processed_frame, face_count = process_frame(
                    frame,
                    self.model,
                    self.detector,
                    self.model_lock
                )

            if face_count == 0:

                cv2.putText(
                    processed_frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            return VideoFrame.from_ndarray(
                processed_frame,
                format="bgr24"
            )

        except Exception as error:

            print("WEBCAM PROCESSING ERROR:", error)

            return VideoFrame.from_ndarray(
                frame,
                format="bgr24"
            )


# =========================================================
# MAIN APPLICATION
# =========================================================

st.title("Facial Emotion Detection")
st.write(
    "Detect faces and recognize facial expressions "
    "using a trained deep learning model."
)

try:

    model = load_emotion_model()

    st.success("V4 emotion detection model loaded successfully.")

except Exception as error:

    st.error(f"Model loading error: {error}")
    st.stop()


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "Live Webcam Detection",
    "Upload Image Detection"
])


# =========================================================
# TAB 1: LIVE WEBCAM
# =========================================================

with tab1:

    st.subheader("Real-Time Facial Emotion Detection")

    st.write(
        "Allow camera access and position your face "
        "in front of the camera."
    )

    webrtc_streamer(
        key="facial-emotion-v4",
        video_processor_factory=lambda: EmotionVideoProcessor(model),
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


# =========================================================
# TAB 2: IMAGE UPLOAD
# =========================================================

with tab2:

    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        image_array = np.array(image)

        frame = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "Detect Face and Emotion",
            type="primary"
        ):

            try:

                detector = create_face_detector()

                height, width = frame.shape[:2]

                detector.setInputSize((width, height))

                _, faces = detector.detect(frame)

                if faces is None or len(faces) == 0:

                    st.warning(
                        "No face detected. Please upload a clear "
                        "front-facing image."
                    )

                else:

                    result_frame = frame.copy()

                    results = []

                    for face in faces:

                        x, y, w, h = [
                            int(value) for value in face[:4]
                        ]

                        x = max(0, x)
                        y = max(0, y)

                        x2 = min(width, x + w)
                        y2 = min(height, y + h)

                        face_crop = result_frame[y:y2, x:x2]

                        if face_crop.size == 0:
                            continue

                        emotion, confidence, predictions = (
                            predict_emotion(face_crop, model)
                        )

                        results.append(
                            (emotion, confidence, predictions)
                        )

                        result_frame = draw_emotion(
                            result_frame,
                            face,
                            model
                        )

                    if not results:

                        st.warning("No valid face could be processed.")

                    else:

                        st.success(
                            f"Detected {len(results)} face(s)."
                        )

                        result_rgb = cv2.cvtColor(
                            result_frame,
                            cv2.COLOR_BGR2RGB
                        )

                        st.image(
                            result_rgb,
                            caption="Face and Emotion Detection Result",
                            use_container_width=True
                        )

                        for index, result in enumerate(results):

                            emotion, confidence, predictions = result

                            st.markdown(
                                f"### Face {index + 1}: {emotion}"
                            )

                            st.metric(
                                "Prediction Confidence",
                                f"{confidence:.2f}%"
                            )

                            st.write("Emotion probabilities:")

                            for emotion_name, probability in zip(
                                EMOTIONS,
                                predictions
                            ):

                                st.write(
                                    f"{emotion_name}: "
                                    f"{probability * 100:.2f}%"
                                )

                                st.progress(
                                    float(np.clip(probability, 0, 1))
                                )

            except Exception as error:

                st.error(
                    f"Image processing error: {error}"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Facial Emotion Detection | CNN V4 + FER+ | "
    "OpenCV YuNet Face Detection"
)
