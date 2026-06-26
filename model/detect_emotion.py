import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from db import connection,cursor
# Get current folder
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model_path = os.path.join(script_dir, "emotion_model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = load_model(model_path)

# Load face detector
cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Could not load haarcascade_frontalface_default.xml")
    exit()

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{emotion} ({confidence:.1f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        sql = """
        INSERT INTO emotion_logs (emotion, confidence)
        VALUES (%s, %s)
        """

        values = (emotion, float(confidence))

        cursor.execute(sql, values)
        connection.commit()

    cv2.imshow("Human Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()