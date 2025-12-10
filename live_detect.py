import cv2
import numpy as np
import tensorflow as tf
from collections import deque

MODEL_PATH = "models/crime_detection_model.keras"
FRAME_COUNT = 30
IMAGE_SIZE = 128

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Store last 30 frames
frame_buffer = deque(maxlen=FRAME_COUNT)

# Start webcam
cap = cv2.VideoCapture(0)

print("Webcam started... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display window size adjustment
    display_frame = cv2.resize(frame, (640, 480))

    # Preprocess frame
    resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    resized = resized / 255.0

    frame_buffer.append(resized)

    label = "Collecting frames..."
    color = (0, 255, 255)

    # When 30 frames collected → predict
    if len(frame_buffer) == FRAME_COUNT:
        input_frames = np.array(frame_buffer).reshape(1, FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE, 3)
        pred = model.predict(input_frames)[0][0]

        if pred > 0.5:
            label = f"CRIME ALERT ({pred:.2f})"
            color = (0, 0, 255)   # Red
        else:
            label = f"Normal ({pred:.2f})"
            color = (0, 255, 0)   # Green

    # Display label on screen
    cv2.putText(display_frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2)

    cv2.imshow("Live Crime Detection", display_frame)

    # Quit if 'Q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

