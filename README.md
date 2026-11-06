# Facial Emotion Detection Using Deep Learning

## Project Overview

A real-time facial emotion detection application that detects human faces and predicts facial expressions using a Convolutional Neural Network (CNN).

The application supports live webcam detection and image upload detection.

## Features

- Real-time face detection using OpenCV YuNet.
- Facial emotion classification using a trained CNN model.
- Live webcam emotion detection.
- Image upload and emotion prediction.
- Confidence scores for predicted emotions.
- Emotion probability display.

## Emotion Classes

The V4 model predicts eight emotion classes:

1. Neutral
2. Happiness
3. Surprise
4. Sadness
5. Anger
6. Disgust
7. Fear
8. Contempt

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- Streamlit WebRTC
- NumPy
- Pillow

## Project Structure

```text
Facial Emotion Detection/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── face_detection_yunet_2023mar.onnx
├── models_v4/
│   ├── best_v4.keras
│   ├── final_v4.keras
│   ├── backup/
│   └── evaluation/
├── dataset/
├── ferplus_data/
├── archive/
└── venv/
```

## Installation

### 1. Clone or download the project

Open the project directory in VS Code.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

## Model Information

- Model version: V4
- Architecture: Custom CNN
- Input image size: 48 × 48 grayscale
- Output classes: 8
- Face detector: OpenCV YuNet
- Model file: `models_v4/best_v4.keras`

## Important Notes

- Allow webcam access in your browser for live detection.
- Use clear, front-facing images with adequate lighting.
- Emotion predictions are model estimates and may not always reflect a person's actual emotional state.
- Performance varies across emotion classes.

## Project Status

The V4 model is integrated with the Streamlit application. Live webcam detection and image upload detection are implemented.

## License

Add the appropriate license before publicly distributing this project.
