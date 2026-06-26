AI Human Emotion Detection System

📌 Project Overview

The AI Human Emotion Detection System is a deep learning application that detects human emotions in real time using a webcam. It uses OpenCV for face detection and a Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify facial expressions into seven emotions.

🎯 Features

- Real-time webcam emotion detection
- Face detection using OpenCV
- Emotion classification using a CNN model
- Detects seven emotions:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
- Confidence score display
- MySQL database integration for storing predictions
- Easy-to-use interface

🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- MySQL
- FastAPI (Backend)
- (Frontend)
- Git & GitHub

📂 Project Structure

Emotion-detection/
│
├── backend/
├── dataset/
├── frontend/
├── model/
│   ├── detect_emotion.py
│   ├── train_model.py
│   ├── emotion_model.h5
│   ├── db.py
│   ├── labels.txt
│   └── haarcascade_frontalface_default.xml
│
├── requirements.txt
├── README.md

📊 Dataset

Dataset: FER-2013

The model is trained to classify facial expressions into seven emotion categories using grayscale facial images.

⚙️ Installation

Clone the repository:

git clone https://github.com/hima065/Emotion-Detection.git

Go to the project folder:

cd Emotion-detection

Install the required packages:

pip install -r requirements.txt

▶️ Run the Project

Run the application:

python model/detect_emotion.py

📈 Future Enhancements

- User authentication
- Emotion analytics dashboard
- PDF and CSV report generation
- Email notifications
- Cloud deployment
- Mobile application integration
