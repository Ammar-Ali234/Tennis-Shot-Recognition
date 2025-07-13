import os
import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
input_dir = 'Videos'  # top folder with class-named subfolders
output_dir = 'pose_csvs'
model = YOLO('yolo11n-pose.pt')  # or yolov8s-pose.pt

sequence_length = 30  # number of frames to keep per sequencess

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

classes = os.listdir(input_dir)

for cls in classes:
    class_path = os.path.join(input_dir, cls)
    if not os.path.isdir(class_path):
        continue

    save_class_path = os.path.join(output_dir, cls)
    os.makedirs(save_class_path, exist_ok=True)

    for video_file in os.listdir(class_path):
        if not video_file.endswith(('.mp4', '.mov', '.avi')):
            continue

        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)

        keypoints_sequence = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 1 == 0:  # use every frame
                results = model(frame)

                # Get first person only
                keypoints = results[0].keypoints
                if keypoints is not None and len(keypoints.xy) > 0:
                    person_keypoints = keypoints.xy[0].cpu().numpy().flatten()  # (num_keypoints * 2,)
                    keypoints_sequence.append(person_keypoints)
                else:
                    # If no person found, append zeros
                    keypoints_sequence.append(np.zeros(33 * 2))

            frame_idx += 1

        cap.release()

        if len(keypoints_sequence) >= sequence_length:
            keypoints_sequence = keypoints_sequence[:sequence_length]
            csv_name = os.path.splitext(video_file)[0] + '.csv'
            save_path = os.path.join(save_class_path, csv_name)
            np.savetxt(save_path, keypoints_sequence, delimiter=',')
            print(f"Saved: {save_path}")
        else:
            print(f"Skipped: {video_file} (not enough frames)")

print("✅ All pose CSVs generated.")
