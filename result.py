import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/crime_detection_model.keras"
FRAME_COUNT = 30
IMAGE_SIZE = 128

model = tf.keras.models.load_model(MODEL_PATH)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < FRAME_COUNT:
        print("Video too short!")
        return None

    step = total // FRAME_COUNT

    for i in range(FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    return np.array(frames)

def predict_video(video_path):
    frames = extract_frames(video_path)
    if frames is None:
        return

    frames = np.expand_dims(frames, axis=0)

    pred = model.predict(frames)[0][0]

    if pred > 0.5:
        print(f"Prediction: CRIME ({pred:.2f})")
    else:
        print(f"Prediction: NORMAL ({pred:.2f})")

# -----------------------------
# CHANGE THIS TO THE VIDEO YOU WANT TO TEST
# -----------------------------
predict_video("dataset/crime/video1.mp4")
