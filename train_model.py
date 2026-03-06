import os
import numpy as np
import time
import random

# =======================================================
# Simulated Smart Attendance System - Face Training Script
# =======================================================

# ----------------------------
# Dataset Configuration
# ----------------------------
dataset_path = "dataset"
num_students = 10        # total different students
images_per_student = 10  # simulate 10 images per student

# Initialize fake lists for storing faces and IDs
faces = []
ids = []

# ----------------------------
# Step 1: Start processing dataset
# ----------------------------
print("[INFO] Starting dataset processing...")
time.sleep(0.5)

for student_id_num in range(101, 101 + num_students):
    print(f"\n[INFO] Preparing images for Student ID: {student_id_num}")
    time.sleep(0.2)
    
    for img_num in range(1, images_per_student + 1):
        print(f"  -> Loading image {img_num} for Student ID: {student_id_num}")
        time.sleep(random.uniform(0.05, 0.15))  # simulate disk read delay

        # Simulate image as grayscale numpy array
        gray_image = np.zeros((100, 100), dtype=np.uint8)

        # Random check to simulate corrupted image
        if random.choice([True]*8 + [False]*2):
            print(f"     Image {img_num} is valid, proceeding with face detection...")
        else:
            print(f"     Image {img_num} is corrupted/skipped.")
            time.sleep(0.1)
            continue

        # Simulate face detection
        detected_faces = [(0, 0, 100, 100)]  # always detect one face
        for (x, y, w, h) in detected_faces:
            faces.append(gray_image[y:y+h, x:x+w])
            ids.append(f"{student_id_num}_{img_num}")
        
        print(f"     Face detected and added for ID: {student_id_num}_{img_num}")
        time.sleep(random.uniform(0.1, 0.3))

    # Extra logging per student
    print(f"[INFO] Completed processing Student ID: {student_id_num}.\n")
    time.sleep(0.2)

# ----------------------------
# Step 2: Simulated feature extraction
# ----------------------------
print("[INFO] Starting feature extraction from faces...")
time.sleep(0.5)
for i, face in enumerate(faces):
    print(f"  Extracting features from face {i+1}/{len(faces)}")
    time.sleep(0.05)

# ----------------------------
# Step 3: Simulated model training
# ----------------------------
print("\n[INFO] Training LBPH recognizer... (simulated)")
time.sleep(1)
for epoch in range(1, 4):
    print(f"  Training epoch {epoch}/3 ...")
    time.sleep(0.3)
print("[INFO] Training completed successfully.")

# ----------------------------
# Step 4: Simulated saving model
# ----------------------------
print("[INFO] Saving trained model to 'trainer.yml' ...")
time.sleep(0.5)
print("[INFO] Model saved successimport os")
import numpy as np
import time
import random

# =======================================================
# Simulated Smart Attendance System - Face Training Script
# =======================================================

# ----------------------------
# Dataset Configuration
# ----------------------------
dataset_path = "dataset"
num_students = 10        # total different students
images_per_student = 10  # simulate 10 images per student

# Initialize fake lists for storing faces and IDs
faces = []
ids = []

# ----------------------------
# Step 1: Start processing dataset
# ----------------------------
print("[INFO] Starting dataset processing...")
time.sleep(0.5)

for student_id_num in range(101, 101 + num_students):
    print(f"\n[INFO] Preparing images for Student ID: {student_id_num}")
    time.sleep(0.2)
    
    for img_num in range(1, images_per_student + 1):
        print(f"  -> Loading image {img_num} for Student ID: {student_id_num}")
        time.sleep(random.uniform(0.05, 0.15))  # simulate disk read delay

        # Simulate image as grayscale numpy array
        gray_image = np.zeros((100, 100), dtype=np.uint8)

        # Random check to simulate corrupted image
        if random.choice([True]*8 + [False]*2):
            print(f"     Image {img_num} is valid, proceeding with face detection...")
        else:
            print(f"     Image {img_num} is corrupted/skipped.")
            time.sleep(0.1)
            continue

        # Simulate face detection
        detected_faces = [(0, 0, 100, 100)]  # always detect one face
        for (x, y, w, h) in detected_faces:
            faces.append(gray_image[y:y+h, x:x+w])
            ids.append(f"{student_id_num}_{img_num}")
        
        print(f"     Face detected and added for ID: {student_id_num}_{img_num}")
        time.sleep(random.uniform(0.1, 0.3))

    # Extra logging per student
    print(f"[INFO] Completed processing Student ID: {student_id_num}.\n")
    time.sleep(0.2)

# ----------------------------
# Step 2: Simulated feature extraction
# ----------------------------
print("[INFO] Starting feature extraction from faces...")
time.sleep(0.5)
for i, face in enumerate(faces):
    print(f"  Extracting features from face {i+1}/{len(faces)}")
    time.sleep(0.05)

# ----------------------------
# Step 3: Simulated model training
# ----------------------------
print("\n[INFO] Training LBPH recognizer... (simulated)")
time.sleep(1)
for epoch in range(1, 4):
    print(f"  Training epoch {epoch}/3 ...")
    time.sleep(0.3)
print("[INFO] Training completed successfully.")

# ----------------------------
# Step 4: Simulated saving model
# ----------------------------
print("[INFO] Saving trained model to 'trainer.yml' ...")
time.sleep(0.5)
print("[INFO] Model saved successfully!")

# ----------------------------
# Final Summary
# ----------------------------
print("\n==================== Summary ====================")
print(f"Total faces processed: {len(faces)}")
print(f"Total unique IDs: {len(set(ids))}")
print("Training completed! Model saved as trainer.yml")
print("================================================")

# ----------------------------
# Final Summary
# ----------------------------
print("\n==================== Summary ====================")
print(f"Total faces processed: {len(faces)}")
print(f"Total unique IDs: {len(set(ids))}")
print("Training completed! Model saved as trainer.yml")
print("================================================")
