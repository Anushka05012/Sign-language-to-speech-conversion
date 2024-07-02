import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

# Load the current index for each class from a file
index_file = 'index.txt'
current_index = [0] * number_of_classes
if os.path.exists(index_file):
    with open(index_file, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            current_index[i] = int(line.strip())

cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Mirror the camera feed
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)

try:
    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print('Collecting data for class {}'.format(j))

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
            cv2.putText(frame, 'Collect images. Press "Q".', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(25)
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("Shutting down the camera.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        counter = current_index[j]  # Start from the current index
        while counter < current_index[j] + dataset_size:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
            cv2.imshow('frame', frame)
            key = cv2.waitKey(25)
            if key == ord('s'):
                print("Shutting down the camera.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

            counter += 1

        # Save the current index back to the file
        current_index[j] = counter
        with open(index_file, 'w') as file:
            for index in current_index:
                file.write(str(index) + '\n')

finally:
    cap.release()
    cv2.destroyAllWindows()
