import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

dataset_path = "dataset"
if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    exit()

X, y = [], []
student_image_count = {}

print("Loading dataset images...")

# Load images and labels
for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        parts = file.split(".")  # User.ID.count.jpg
        if len(parts) >= 3:
            student_id = parts[1]

            img_path = os.path.join(dataset_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read {file}")
                continue

            img_resized = cv2.resize(img, (100,100))  # uniform size
            X.append(img_resized.flatten())
            y.append(student_id)

            # Count images per student
            if student_id not in student_image_count:
                student_image_count[student_id] = 0
            student_image_count[student_id] += 1

# Check dataset
if len(X) == 0:
    print("No images found in dataset. Exiting...")
    exit()

X = np.array(X)
y = np.array(y)

print("\nImage count per student:")
for student, count in student_image_count.items():
    print(f"Student {student}: {count} images")

# Normalize pixel values
X = X / 255.0

# Train KNN classifier
print("\nTraining KNN classifier...")
model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
model.fit(X, y)

print("Training completed successfully!")
print(f"Number of students trained: {len(student_image_count)}")
print(f"Number of images used: {len(X)}")

# Save model
model_file = "face_model.pkl"
with open(model_file, "wb") as f:
    pickle.dump(model, f)

print(f"Trained model saved to '{model_file}'")