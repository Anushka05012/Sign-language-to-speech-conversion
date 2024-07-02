#Data Set
import os
import pickle

import mediapipe as mp
import cv2

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Check if the data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory {DATA_DIR} does not exist.")
    exit()

# Process each class directory
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    # Process each image in the class directory
    for img_path in os.listdir(class_dir):
        img_full_path = os.path.join(class_dir, img_path)
        img = cv2.imread(img_full_path)

        if img is None:
            print(f"Error: Unable to read image {img_full_path}.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                data_aux = []

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                min_x, min_y = min(x_), min(y_)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

                data.append(data_aux)
                labels.append(int(dir_))  # Assuming the directory name is the label

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Release resources
hands.close()

print("Data processing completed successfully!")
