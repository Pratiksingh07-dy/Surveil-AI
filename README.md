🚨 Surveil-AI: Real-Time Crime Detection
📌 1. Project Overview

Surveil-AI is a deep-learning–powered surveillance system that detects violent or suspicious activity from videos and real-time webcam footage.
The model combines CNN (for visual features) and LSTM (for motion learning) to classify footage as Crime or Normal.

⭐ 2. Features
🎥 Real-time webcam crime detection
📁 Crime prediction on saved video files
🧠 Trainable on custom datasets
⚡ Fast inference using MobileNetV2
📊 Confidence score output

🧠 3. How It Works (Model Architecture)
The system processes each video into 30 frames, then:

1. MobileNetV2 (CNN Backbone)
Extracts visual features from every frame.

2. TimeDistributed Layer
Applies MobileNetV2 to each frame sequentially.

3. LSTM Layer
Learns motion patterns across frames.

4. Dense Classifier
Outputs a probability score:
0 → Normal
1 → Crime

This hybrid CNN + LSTM architecture makes the model effective for activity recognition.

▶️ 4. How to Run the Project
Activate virtual environment
venv\Scripts\activate

Train the model
python train.py

Run prediction on a video
python result.py

Start real-time webcam detection
python live_detect.py

Press Q to stop the webcam.

📂 5. Dataset Structure
dataset/
   crime/
      video1.mp4
      video2.mp4
   normal/
      video3.mp4
      video4.mp4


✔ Each video must have ≥ 30 frames
✔ Supported formats: mp4, avi, mov

📦 6. Requirements
tensorflow==2.12.0
tensorflow-intel==2.12.0
tensorflow-estimator==2.12.0
tensorflow-io-gcs-filesystem==0.31.0

numpy==1.23.5
opencv-python==4.7.0.72
Pillow==10.3.0
scikit-learn==1.3.2
matplotlib==3.7.5

gast==0.4.0
keras==2.12.0
protobuf==4.25.8
wrapt==1.14.2
h5py==3.11.0
