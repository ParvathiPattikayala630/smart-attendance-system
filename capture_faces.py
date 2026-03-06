import cv2
import os

# Create dataset folder if not exists
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

student_id = input("Enter Student ID: ")

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
print("Look at the camera. Capturing images...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x,y,w,h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        file_name = f"{dataset_path}/User.{student_id}.{count}.jpg"
        cv2.imwrite(file_name, face_img)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == 13 or count >= 30:  # Press Enter or capture 30 images
        break

cap.release()
cv2.destroyAllWindows()
print("Face capture completed!")