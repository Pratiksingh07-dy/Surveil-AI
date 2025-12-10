import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import LSTM, Dense, GlobalAveragePooling2D, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

# ------------------------------
# SETTINGS
# ------------------------------
DATASET_PATH = "dataset"
FRAME_COUNT = 30          # frames per video
IMAGE_SIZE = 128          # resize each frame
BATCH_SIZE = 4

# ------------------------------
# FUNCTION TO EXTRACT FRAMES
# ------------------------------
def extract_frames(video_path, frame_count=FRAME_COUNT):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < frame_count:
        return None  

    step = total_frames // frame_count
    
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == frame_count:
        return np.array(frames)
    return None

# ------------------------------
# LOAD DATASET
# ------------------------------
def load_dataset():
    X, y = [], []

    classes = {"normal": 0, "crime": 1}

    for label in classes:
        folder = os.path.join(DATASET_PATH, label)

        for video_name in os.listdir(folder):
            video_path = os.path.join(folder, video_name)
            frames = extract_frames(video_path)

            if frames is not None:
                X.append(frames)
                y.append(classes[label])
                print(f"Loaded {video_name} → {label}")

    X = np.array(X)
    y = np.array(y)

    print("Dataset loaded:")
    print("Videos:", len(X))
    print("Shape:", X.shape)
    return X, y

# ------------------------------
# BUILD MODEL (CNN + LSTM)
# ------------------------------
def build_model():
    cnn = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    cnn.trainable = False  # freeze CNN

    model = Sequential([
        TimeDistributed(cnn, input_shape=(FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

# ------------------------------
# TRAIN MODEL
# ------------------------------
X, y = load_dataset()

model = build_model()

checkpoint = ModelCheckpoint("models/crime_detection_model.keras",
                             monitor='loss',
                             save_best_only=True,
                             mode='min')

model.fit(X, y, epochs=10, batch_size=BATCH_SIZE, callbacks=[checkpoint])

print("Training complete! Model saved at models/crime_detection_model.keras")

