import cv2
import os

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

student_id = input("Enter Student ID: ")

path = "dataset"
if not os.path.exists(path):
    os.makedirs(path)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        file_name = f"{path}/User.{student_id}.{count}.jpg"
        cv2.imwrite(file_name, face)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Face Capture", frame)

    if count >= 10:
        break

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

print("Face images captured successfully!")
