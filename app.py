# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# import torch
# from collections import deque
# from ultralytics import YOLO
# from PIL import Image

# # ========== CONFIG ==========
# POSE_MODEL_PATH = 'yolo11n-pose.pt'
# LSTM_MODEL_PATH = 'best_lstm_model_fg.pth'
# SEQUENCE_LENGTH = 30
# FEATURE_DIM = 34
# LABELS = ['backhand', 'forehand', 'serve']
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# # ============================

# # ---- LSTM Model ----
# class LSTMClassifier(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super().__init__()
#         self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.dropout = torch.nn.Dropout(0.3)
#         self.fc = torch.nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
#         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.dropout(out[:, -1, :])
#         return self.fc(out)

# # ---- Load Models ----
# pose_detector = YOLO(POSE_MODEL_PATH)
# lstm_model = LSTMClassifier(FEATURE_DIM, 128, 2, len(LABELS)).to(DEVICE)
# lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
# lstm_model.eval()

# # ---- UI ----
# st.title("🎾 Shot Classification App")
# input_mode = st.radio("Choose input source:", ["Upload Video", "Webcam"])

# buffer = deque(maxlen=SEQUENCE_LENGTH)

# def process_frame(frame):
#     results = pose_detector(frame)[0]

#     if results.keypoints is not None and len(results.keypoints.xy) > 0:
#         kpts = results.keypoints.xy[0].cpu().numpy().reshape(-1)
#     else:
#         kpts = np.zeros(FEATURE_DIM, dtype=float)

#     buffer.append(kpts)

#     pred_label = ""
#     confidence = 0
#     if len(buffer) == SEQUENCE_LENGTH:
#         seq = np.stack(buffer)
#         seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             logits = lstm_model(seq_tensor)
#             pred = torch.softmax(logits, dim=1)
#             pred_idx = pred.argmax(dim=1).item()
#             pred_label = LABELS[pred_idx]
#             confidence = pred[0, pred_idx].item()

#     display = results.plot()
#     if pred_label:
#         cv2.putText(display, f"{pred_label} ({confidence:.2f})", (10, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     return display

# # ---- Webcam Mode ----
# if input_mode == "Webcam":
#     run = st.checkbox("Start Webcam")
#     FRAME_WINDOW = st.image([])

#     cap = cv2.VideoCapture(0)

#     while run:
#         ret, frame = cap.read()
#         if not ret:
#             st.warning("Could not access webcam.")
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         processed = process_frame(frame)
#         FRAME_WINDOW.image(processed, channels="RGB")

#     cap.release()

# # ---- Video Upload Mode ----
# else:
#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
#     if uploaded_file:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())

#         cap = cv2.VideoCapture(tfile.name)
#         FRAME_WINDOW = st.image([])

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             processed = process_frame(frame)
#             FRAME_WINDOW.image(processed, channels="RGB")

#         cap.release()
#         st.success("✅ Finished processing video.")


import os
import time
import cv2
import numpy as np
import tempfile
import streamlit as st
import torch
from collections import deque
from ultralytics import YOLO

# ========== CONFIG ==========
POSE_MODEL_PATH = 'yolo11n-pose.pt'
LSTM_MODEL_PATH = 'best_lstm_model_fg.pth'
SEQUENCE_LENGTH = 30
FEATURE_DIM = 34
LABELS = ['backhand', 'forehand', 'serve']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ============================

# ---- LSTM Model ----
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# ---- Load Models ----
pose_detector = YOLO(POSE_MODEL_PATH)
lstm_model = LSTMClassifier(FEATURE_DIM, 128, 2, len(LABELS)).to(DEVICE)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
lstm_model.eval()

# ---- Streamlit UI ----
st.set_page_config(page_title="Tennis Shot Classifier", layout="centered")
st.title("🎾 Tennis Shot Classification")

tab1, tab2 = st.tabs(["📹 Webcam", "📁 Upload Video"])

buffer = deque(maxlen=SEQUENCE_LENGTH)

def draw_text_box(img, text, position, box_color, text_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(img, (x, y - 25), (x + text_size[0] + 10, y + 10), box_color, -1)
    cv2.putText(img, text, (x + 5, y), font, scale, text_color, thickness)

def process_frame(frame, show_fps=False, fps=None):
    results = pose_detector(frame)[0]

    if results.keypoints is not None and len(results.keypoints.xy) > 0:
        kpts = results.keypoints.xy[0].cpu().numpy().reshape(-1)
    else:
        kpts = np.zeros(FEATURE_DIM, dtype=float)

    buffer.append(kpts)

    pred_label = ""
    confidence = 0
    if len(buffer) == SEQUENCE_LENGTH:
        seq = np.stack(buffer)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = lstm_model(seq_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            pred_label = LABELS[pred_idx]
            confidence = probs[0, pred_idx].item()

    display = results.plot()

    # Draw FPS (top-left) and prediction (top-right)
    h, w = display.shape[:2]
    if show_fps and fps:
        draw_text_box(display, f"{fps:.1f} FPS", (10, 30), (255, 255, 255), (0, 0, 255))
    if pred_label:
        draw_text_box(display, f"{pred_label} ({confidence:.2f})", (w - 240, 30), (255, 255, 255), (0, 0, 255))

    return display

# ---- Webcam Tab ----
with tab1:
    run = st.checkbox("📷 Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("🚫 Could not access webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            processed = process_frame(frame, show_fps=True, fps=fps)
            FRAME_WINDOW.image(processed, channels="RGB")
    else:
        st.info("✔️ Click the checkbox to start webcam.")

# ---- Upload Video Tab ----
with tab2:
    uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            processed = process_frame(frame, show_fps=True, fps=fps)
            FRAME_WINDOW.image(processed, channels="RGB")

        cap.release()
        st.success("✅ Finished processing video.")
