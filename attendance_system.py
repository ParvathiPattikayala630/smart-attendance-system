import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

# Load trained model
with open("face_model.pkl", "rb") as f:
    model = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

attendance = {}
dataset_ids = []  # List of all student IDs
for file in os.listdir("dataset"):
    if file.endswith(".jpg"):
        student_id = file.split(".")[1]
        if student_id not in dataset_ids:
            dataset_ids.append(student_id)

print("Attendance system started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x,y,w,h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_small = cv2.resize(face_img, (100,100)).flatten()
        pred = model.predict([face_small])[0]
        dist, _ = model.kneighbors([face_small], n_neighbors=1)
        label = f"Known ✅ ({pred})" if dist[0][0] < 5000 else "Unknown ❌"  # threshold

        if "Known" in label:
            attendance[pred] = "Present"

        # Draw rectangle and label
        color = (0,255,0) if "Known" in label else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance System", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Save attendance CSV
all_students = dataset_ids
df = pd.DataFrame({
    "Name": all_students,
    "Status": ["Present" if s in attendance else "Absent" for s in all_students]
})
df.to_csv("attendance.csv", index=False)
print("Attendance saved to 'attendance.csv'.")

cap.release()
cv2.destroyAllWindows()