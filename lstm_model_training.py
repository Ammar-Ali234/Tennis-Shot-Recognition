# import os
# import re
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

# # ========== CONFIG ==========
# POSE_CSV_DIR   = 'pose_csvs'
# SEQUENCE_LENGTH = 30
# BATCH_SIZE      = 16
# EPOCHS          = 100
# LEARNING_RATE   = 0.001
# # ============================

# class PoseDataset(Dataset):
#     def __init__(self, csv_root):
#         self.samples = []
#         self.labels  = []
#         self.label_encoder = LabelEncoder()

#         # regex to pull leading letters as the label
#         label_re = re.compile(r'^([A-Za-z]+)')
#         for fname in sorted(os.listdir(csv_root)):
#             if not fname.lower().endswith('.csv'):
#                 continue
#             m = label_re.match(fname)
#             if not m:
#                 continue
#             label = m.group(1).lower()  # e.g. "backhand" or "forehand"
#             path  = os.path.join(csv_root, fname)
#             arr   = np.loadtxt(path, delimiter=',')
#             # ensure correct sequence length
#             if arr.shape[0] != SEQUENCE_LENGTH:
#                 print(f"Skipping {fname}: expected {SEQUENCE_LENGTH} rows, got {arr.shape[0]}")
#                 continue
#             self.samples.append(arr)
#             self.labels .append(label)

#         if len(self.samples) == 0:
#             raise RuntimeError(f"No valid samples found in {csv_root}.")

#         # stack into array: (N, T, D)
#         self.samples = np.stack(self.samples)
#         # encode labels to 0…C-1
#         self.labels  = self.label_encoder.fit_transform(self.labels)
#         print(f"Loaded {len(self.samples)} sequences, feature dim = {self.samples.shape[2]}, "
#               f"{len(self.label_encoder.classes_)} classes: {self.label_encoder.classes_}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         X = torch.tensor(self.samples[idx], dtype=torch.float32)
#         y = torch.tensor(self.labels[idx], dtype=torch.long)
#         return X, y


# class LSTMClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super().__init__()
#         self.lstm    = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(0.3)
#         self.fc      = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         # auto-initialize hidden/cell based on input shape
#         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
#         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.dropout(out[:, -1, :])
#         return self.fc(out)


# def train():
#     dataset = PoseDataset(POSE_CSV_DIR)

#     # split
#     X_train, X_val, y_train, y_val = train_test_split(
#         dataset.samples, dataset.labels,
#         test_size=0.2, random_state=42, stratify=dataset.labels)

#     train_loader = DataLoader(
#         list(zip(
#             torch.tensor(X_train, dtype=torch.float32),
#             torch.tensor(y_train, dtype=torch.long)
#         )),
#         batch_size=BATCH_SIZE, shuffle=True
#     )
#     val_loader = DataLoader(
#         list(zip(
#             torch.tensor(X_val, dtype=torch.float32),
#             torch.tensor(y_val, dtype=torch.long)
#         )),
#         batch_size=BATCH_SIZE
#     )

#     # infer input_size dynamically
#     input_size  = dataset.samples.shape[2]
#     num_classes = len(dataset.label_encoder.classes_)
#     model = LSTMClassifier(input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model  = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     for epoch in range(1, EPOCHS+1):
#         model.train()
#         total_loss, correct, total = 0, 0, 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss    = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             preds = outputs.argmax(dim=1)
#             correct  += (preds == labels).sum().item()
#             total    += labels.size(0)

#         train_acc = 100 * correct/total
#         print(f"Epoch {epoch}/{EPOCHS} — Loss: {total_loss:.4f}, Acc: {train_acc:.2f}%")

#     torch.save(model.state_dict(), 'lstm_pose_classifier_s1.pth')
#     print("✅ Model trained and saved.")

# if __name__ == "__main__":
#     train()


import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
import shutil

# ========== CONFIG ==========
POSE_CSV_DIR = 'pose_csvs'
AUGMENTED_CSV_DIR = 'pose_csvs_augmented'
SEQUENCE_LENGTH = 30
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
# ============================

# ----------- AUGMENTATION FUNCTIONS -----------
def augment_sequence(sequence):
    augmented = []

    # 1. Horizontal flip (swap left/right keypoints)
    flipped = sequence.copy()
    # Assuming COCO keypoints order - update this swap logic if different
    left_indices  = [5, 7, 9, 11, 13, 15]
    right_indices = [6, 8,10, 12, 14, 16]
    for l, r in zip(left_indices, right_indices):
        flipped[:, [l*2, r*2]] = flipped[:, [r*2, l*2]]  # x-coords
        flipped[:, [l*2+1, r*2+1]] = flipped[:, [r*2+1, l*2+1]]  # y-coords
    augmented.append(flipped)

    # 2. Add Gaussian noise
    noisy = sequence + np.random.normal(0, 2, size=sequence.shape)
    augmented.append(noisy)

    # 3. Slight frame jitter
    jittered = sequence.copy()
    jitter_indices = np.random.permutation(SEQUENCE_LENGTH)
    jittered = jittered[jitter_indices[:SEQUENCE_LENGTH]]
    augmented.append(jittered)

    return augmented

def create_augmented_dataset(input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.csv'):
            continue
        src_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]
        arr = np.loadtxt(src_path, delimiter=',')
        if arr.shape[0] != SEQUENCE_LENGTH:
            print(f"Skipping {fname} due to length mismatch.")
            continue

        # Save original
        np.savetxt(os.path.join(output_dir, f"{base_name}_orig.csv"), arr, delimiter=',')

        # Generate and save augmentations
        aug_seqs = augment_sequence(arr)
        for i, aug in enumerate(aug_seqs):
            out_fname = os.path.join(output_dir, f"{base_name}_aug{i+1}.csv")
            np.savetxt(out_fname, aug, delimiter=',')

    print(f"✅ Augmented CSVs saved to: {output_dir}")
# -----------------------------------------------

class PoseDataset(Dataset):
    def __init__(self, csv_root):
        self.samples = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        label_re = re.compile(r'^([a-zA-Z]+)')

        for fname in sorted(os.listdir(csv_root)):
            if not fname.endswith('.csv'):
                continue
            m = label_re.match(fname)
            if not m:
                continue
            label = m.group(1).lower()
            path = os.path.join(csv_root, fname)
            arr = np.loadtxt(path, delimiter=',')
            if arr.shape[0] != SEQUENCE_LENGTH:
                continue
            self.samples.append(arr)
            self.labels.append(label)

        if not self.samples:
            raise RuntimeError("No valid data found.")

        self.samples = np.stack(self.samples)
        self.labels = self.label_encoder.fit_transform(self.labels)
        print(f"Loaded {len(self.samples)} sequences. Classes: {self.label_encoder.classes_}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return X, y


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def train():
    create_augmented_dataset(POSE_CSV_DIR, AUGMENTED_CSV_DIR)

    dataset = PoseDataset(AUGMENTED_CSV_DIR)
    X_train, X_val, y_train, y_val = train_test_split(
        dataset.samples, dataset.labels,
        test_size=0.2, stratify=dataset.labels, random_state=42
    )

    train_loader = DataLoader(list(zip(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long))),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(list(zip(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long))),
        batch_size=BATCH_SIZE
    )

    input_size = dataset.samples.shape[2]
    num_classes = len(dataset.label_encoder.classes_)
    model = LSTMClassifier(input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch}/{EPOCHS} — Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm_model_fg.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    print(f"✅ Best validation accuracy: {best_val_acc:.2f}% — model saved to 'best_lstm_model.pth'")


if __name__ == "__main__":
    train()
