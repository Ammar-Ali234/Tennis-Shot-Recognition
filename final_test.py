# # import os
# # import cv2
# # import torch
# # import numpy as np
# # from collections import deque

# # # ========== CONFIG ==========
# # VIDEO_PATH      = 'test.mp4'
# # OUTPUT_PATH     = 'output_annotated.mp4'
# # POSE_MODEL_PATH = 'yolo11n-pose.pt'      # YOLOv8 pose estimation weights
# # LSTM_MODEL_PATH = 'lstm_pose_classifier.pth'
# # SEQUENCE_LENGTH = 30
# # FEATURE_DIM      = 34                  # inferred from your CSVs
# # DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
# # # ============================

# # # 1. Load YOLOv8 pose model (Ultralytics)
# # from ultralytics import YOLO
# # pose_detector = YOLO(POSE_MODEL_PATH)

# # # 2. Load trained LSTM classifier
# # device = torch.device(DEVICE)
# # # adjust input dim if bidirectional etc.
# # class LSTMClassifier(torch.nn.Module):
# #     def __init__(self, input_size, hidden_size, num_layers, num_classes):
# #         super().__init__()
# #         self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
# #                                   batch_first=True, bidirectional=False)
# #         self.dropout = torch.nn.Dropout(0.3)
# #         self.fc = torch.nn.Linear(hidden_size, num_classes)

# #     def forward(self, x):
# #         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
# #         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
# #         out, _ = self.lstm(x, (h0, c0))
# #         out = self.dropout(out[:, -1, :])
# #         return self.fc(out)

# # # instantiate and load state
# # # num_classes = 2  # backhand, forehand\ nmodel = LSTMClassifier(input_size=FEATURE_DIM, hidden_size=128, num_layers=2, num_classes=2).to(device)
# # model = LSTMClassifier(FEATURE_DIM, 128, 2, 2).to(device)
# # model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
# # model.eval()

# # # label mapping
# # LABELS = ['backhand', 'forehand']

# # # 3. Open video
# # cap = cv2.VideoCapture(VIDEO_PATH)
# # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # fps = cap.get(cv2.CAP_PROP_FPS)
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# # # rolling buffer for pose sequences
# # buffer = deque(maxlen=SEQUENCE_LENGTH)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     # run pose detection
# #     results = pose_detector(frame)[0]
# #     # extract keypoints safely
# #     if results.keypoints is not None and len(results.keypoints.xy) > 0:
# #         # get (17,2) coords for first person and flatten
# #         kpts = results.keypoints.xy[0].cpu().numpy().reshape(-1)
# #     else:
# #         # if no detection, zero vector
# #         kpts = np.zeros(FEATURE_DIM, dtype=float)

# #     buffer.append(kpts)

# #     label_text = ''
# #     if len(buffer) == SEQUENCE_LENGTH:
# #         seq = np.stack(buffer)             # (T, D)
# #         seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
# #         with torch.no_grad():
# #             logits = model(seq_tensor)
# #             pred   = logits.argmax(dim=1).item()
# #         label_text = LABELS[pred]

# #     # annotate
# #     display = frame.copy()
# #     if label_text:
# #         cv2.putText(display, f"Shot: {label_text}", (10, 30),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# #     # write frame
# #     out.write(display)
# #     cv2.imshow('Shot Classification', display)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # out.release()
# # cv2.destroyAllWindows()
# # print('✅ Finished processing video. Output saved to', OUTPUT_PATH)('✅ Finished processing video. Output saved to', OUTPUT_PATH)


# import os
# import cv2
# import torch
# import numpy as np
# from collections import deque

# # ========== CONFIG ==========
# VIDEO_PATH      = 'test.mp4'
# OUTPUT_PATH     = 'output_annotated1.mp4'
# POSE_MODEL_PATH = 'yolo11n-pose.pt'      # YOLOv8 pose estimation weights
# LSTM_MODEL_PATH = 'lstm_pose_classifier.pth'
# SEQUENCE_LENGTH = 30
# FEATURE_DIM      = 34                  # inferred from your CSVs
# DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
# # ============================

# # 1. Load YOLOv8 pose model (Ultralytics)
# from ultralytics import YOLO
# pose_detector = YOLO(POSE_MODEL_PATH)

# # 2. Load trained LSTM classifier
# device = torch.device(DEVICE)
# class LSTMClassifier(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super().__init__()
#         self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
#                                   batch_first=True, bidirectional=False)
#         self.dropout = torch.nn.Dropout(0.3)
#         self.fc = torch.nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
#         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.dropout(out[:, -1, :])
#         return self.fc(out)

# model = LSTMClassifier(FEATURE_DIM, 128, 2, 2).to(device)
# model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
# model.eval()

# # label mapping
# LABELS = ['backhand', 'forehand']

# # 3. Open video
# cap = cv2.VideoCapture(VIDEO_PATH)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# buffer = deque(maxlen=SEQUENCE_LENGTH)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # run pose detection
#     results = pose_detector(frame)[0]

#     # extract keypoints
#     if results.keypoints is not None and len(results.keypoints.xy) > 0:
#         kpts = results.keypoints.xy[0].cpu().numpy().reshape(-1)
#     else:
#         kpts = np.zeros(FEATURE_DIM, dtype=float)

#     buffer.append(kpts)

#     label_text = ''
#     if len(buffer) == SEQUENCE_LENGTH:
#         seq = np.stack(buffer)
#         seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
#         with torch.no_grad():
#             logits = model(seq_tensor)
#             pred = logits.argmax(dim=1).item()
#         label_text = LABELS[pred]

#     # get YOLO's color overlay (bbox + skeleton)
#     display = results.plot()

#     # overlay shot label
#     if label_text:
#         cv2.putText(display, f"Shot: {label_text}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # write and show
#     out.write(display)
#     cv2.imshow('Shot Classification', display)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()
# print('✅ Finished processing video. Output saved to', OUTPUT_PATH)


import os
import cv2
import torch
import numpy as np
from collections import deque

# ========== CONFIG ==========
VIDEO_PATH      = 'Videos\\Backhand\\backhand2.mp4'
OUTPUT_PATH     = 'output_annotated4.mp4'
POSE_MODEL_PATH = 'yolo11n-pose.pt'      # YOLOv8 pose estimation weights
LSTM_MODEL_PATH = 'lstm_pose_classifier_s1.pth'
SEQUENCE_LENGTH = 30
FEATURE_DIM      = 34                  # inferred from your CSVs
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
# ============================

# 1. Load YOLOv8 pose model (Ultralytics)
from ultralytics import YOLO
pose_detector = YOLO(POSE_MODEL_PATH)

# 2. Load trained LSTM classifier
device = torch.device(DEVICE)
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=False)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# label mapping
LABELS = ['backhand', 'forehand', 'serve']

# instantiate model with correct number of classes
model = LSTMClassifier(FEATURE_DIM, 128, 2, len(LABELS)).to(device)
model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
model.eval()

# label mapping
LABELS = ['backhand', 'forehand']

# 3. Open video
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

buffer = deque(maxlen=SEQUENCE_LENGTH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run pose detection
    results = pose_detector(frame)[0]

    # extract keypoints
    if results.keypoints is not None and len(results.keypoints.xy) > 0:
        kpts = results.keypoints.xy[0].cpu().numpy().reshape(-1)
    else:
        kpts = np.zeros(FEATURE_DIM, dtype=float)

    buffer.append(kpts)

    label_text = ''
    if len(buffer) == SEQUENCE_LENGTH:
        seq = np.stack(buffer)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(seq_tensor)
            pred = logits.argmax(dim=1).item()
        label_text = LABELS[pred]

    # get YOLO's color overlay (bbox + skeleton)
    display = results.plot()

    # overlay shot label
    if label_text:
        cv2.putText(display, f"Shot: {label_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # write and show
    out.write(display)
    cv2.imshow('Shot Classification', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print('✅ Finished processing video. Output saved to', OUTPUT_PATH)
