import cv2
import os
import numpy as np
from PIL import Image
import time

# --------------------------
# Dataset folder
dataset_path = "dataset"

# Load dataset images
students = {}
for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        student_id = os.path.splitext(file)[0]
        img = Image.open(os.path.join(dataset_path, file)).convert("L").resize((100,100))
        students[student_id] = np.array(img)

print("Dataset loaded:", list(students.keys()))

# --------------------------
# Open webcam
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' or ESC to quit.")

recognized_once = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Take central region (assume face is there)
    h, w = gray.shape
    cx, cy = w//2, h//2
    size = 200
    x1, y1 = max(cx - size//2,0), max(cy - size//2,0)
    x2, y2 = min(cx + size//2, w), min(cy + size//2, h)
    face_roi = gray[y1:y2, x1:x2]
    face_small = cv2.resize(face_roi, (100,100))

    # Compare with dataset
    min_mse = float('inf')
    best_match = None
    for student_id, student_img in students.items():
        mse = np.mean((face_small - student_img)**2)
        if mse < min_mse:
            min_mse = mse
            best_match = student_id

    # Threshold for recognition
    if min_mse < 500:
        label = f"Known ✅ ({best_match})"
    else:
        label = "Unknown ❌"

    # Display on webcam frame
    cv2.putText(frame, label, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if "Known" in label else (0,0,255), 2)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.imshow("Hackathon Attendance Demo", frame)

    # Print only once per person
    if not recognized_once:
        print(label)
        recognized_once = True
        # Pause 3 seconds so faculty can see result
        cv2.waitKey(3000)  

    # Exit on key press
    key = cv2.waitKey(1)
    if key != -1:
        if key & 0xFF == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
