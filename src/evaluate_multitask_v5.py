import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# Paths
# ============================================================

ROOT = "data/processed"

TEST_CSV = os.path.join(ROOT, "test.csv")

LEFT_DIR = os.path.join(
    ROOT, "eyes", "left"
)

RIGHT_DIR = os.path.join(
    ROOT, "eyes", "right"
)

MODEL_PATH = "models/best_multitask_v5.pth"

BATCH_SIZE = 32


# ============================================================
# Labels
# ============================================================

HORIZONTAL = {
    "LEFT": 0,
    "CENTER": 1,
    "RIGHT": 2
}

VERTICAL = {
    "UP": 0,
    "CENTER": 1,
    "DOWN": 2
}

DIRECTION = {
    "UP_LEFT": 0,
    "UP_CENTER": 1,
    "UP_RIGHT": 2,

    "CENTER_LEFT": 3,
    "CENTER_CENTER": 4,
    "CENTER_RIGHT": 5,

    "DOWN_LEFT": 6,
    "DOWN_CENTER": 7,
    "DOWN_RIGHT": 8
}

DIRECTION_NAMES = [
    "UP_LEFT",
    "UP_CENTER",
    "UP_RIGHT",
    "CENTER_LEFT",
    "CENTER_CENTER",
    "CENTER_RIGHT",
    "DOWN_LEFT",
    "DOWN_CENTER",
    "DOWN_RIGHT"
]


# ============================================================
# Dataset
# ============================================================

class GazeDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def load_image(self, path):

        img = Image.open(path).convert("RGB")

        img = np.asarray(
            img,
            dtype=np.float32
        ) / 255.0

        img = torch.from_numpy(img)

        return img.permute(2, 0, 1)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        left = self.load_image(
            os.path.join(
                LEFT_DIR,
                row["left_image"]
            )
        )

        right = self.load_image(
            os.path.join(
                RIGHT_DIR,
                row["right_image"]
            )
        )

        # ----------------------------------------------------
        # Regression targets
        # ----------------------------------------------------

        target_reg = torch.tensor(
            [
                float(row["relative_yaw"]),
                float(row["relative_pitch"])
            ],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Classification targets
        # ----------------------------------------------------

        target_h = HORIZONTAL[
            str(row["target_horizontal"]).upper()
        ]

        target_v = VERTICAL[
            str(row["target_vertical"]).upper()
        ]

        target_d = DIRECTION[
            str(row["target_direction"]).upper()
        ]

        # ----------------------------------------------------
        # Geometry target
        #
        # Normalized pupil position inside each eye crop
        # ----------------------------------------------------

        left_w = (
            float(row["left_crop_xmax"])
            - float(row["left_crop_xmin"])
        )

        left_h = (
            float(row["left_crop_ymax"])
            - float(row["left_crop_ymin"])
        )

        right_w = (
            float(row["right_crop_xmax"])
            - float(row["right_crop_xmin"])
        )

        right_h = (
            float(row["right_crop_ymax"])
            - float(row["right_crop_ymin"])
        )

        left_x = (
            float(row["left_pupil_x"])
            - float(row["left_crop_xmin"])
        ) / left_w

        left_y = (
            float(row["left_pupil_y"])
            - float(row["left_crop_ymin"])
        ) / left_h

        right_x = (
            float(row["right_pupil_x"])
            - float(row["right_crop_xmin"])
        ) / right_w

        right_y = (
            float(row["right_pupil_y"])
            - float(row["right_crop_ymin"])
        ) / right_h

        target_geometry = torch.tensor(
            [
                left_x,
                left_y,
                right_x,
                right_y
            ],
            dtype=torch.float32
        )

        return (
            left,
            right,
            target_reg,
            torch.tensor(target_h, dtype=torch.long),
            torch.tensor(target_v, dtype=torch.long),
            torch.tensor(target_d, dtype=torch.long),
            target_geometry
        )


# ============================================================
# Eye CNN
# ============================================================

class EyeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 16, 3, padding=1
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32, 3, padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, 3, padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64, 64, 3, padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):

        return self.features(x).flatten(1)


# ============================================================
# V5 MODEL
# ============================================================

class MultiTaskGazeV5(nn.Module):

    def __init__(self):

        super().__init__()

        self.eye_cnn = EyeCNN()

        self.shared = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.10)
        )

        # ----------------------------------------------------
        # Gaze regression
        # ----------------------------------------------------

        self.regression_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 2)
        )

        # ----------------------------------------------------
        # Horizontal
        # ----------------------------------------------------

        self.horizontal_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        # ----------------------------------------------------
        # Vertical
        # ----------------------------------------------------

        self.vertical_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        # ----------------------------------------------------
        # Direction fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(102, 64),
            nn.ReLU(),
            nn.Dropout(0.10)
        )

        self.direction_head = nn.Linear(
            64, 9
        )

        # ----------------------------------------------------
        # Geometry head
        #
        # Output:
        # [left_x, left_y, right_x, right_y]
        # ----------------------------------------------------

        self.geometry_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )

    def forward(self, left, right):

        left_features = self.eye_cnn(left)

        right_features = self.eye_cnn(right)

        features = torch.cat(
            [
                left_features,
                right_features
            ],
            dim=1
        )

        features = self.shared(features)

        regression = self.regression_head(
            features
        )

        horizontal = self.horizontal_head(
            features
        )

        vertical = self.vertical_head(
            features
        )

        fusion_input = torch.cat(
            [
                features,
                horizontal,
                vertical
            ],
            dim=1
        )

        fused = self.fusion(
            fusion_input
        )

        direction = self.direction_head(
            fused
        )

        geometry = self.geometry_head(
            features
        )

        return (
            regression,
            horizontal,
            vertical,
            direction,
            geometry
        )


# ============================================================
# Load model
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("U2EYES MULTI-TASK GAZE V5 EVALUATION")
print("=" * 70)

print("Device:", DEVICE)

dataset = GazeDataset(TEST_CSV)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test samples:", len(dataset))


model = MultiTaskGazeV5().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Support common checkpoint formats

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(state_dict)

model.eval()


# ============================================================
# Evaluation storage
# ============================================================

all_reg_true = []
all_reg_pred = []

all_h_true = []
all_h_pred = []

all_v_true = []
all_v_pred = []

all_d_true = []
all_d_pred = []

all_geom_true = []
all_geom_pred = []

all_sample_ids = []


# ============================================================
# Inference
# ============================================================

with torch.no_grad():

    for batch in loader:

        (
            left,
            right,
            target_reg,
            target_h,
            target_v,
            target_d,
            target_geometry
        ) = batch

        left = left.to(DEVICE)
        right = right.to(DEVICE)

        (
            regression,
            horizontal,
            vertical,
            direction,
            geometry
        ) = model(
            left,
            right
        )

        # Regression

        all_reg_true.append(
            target_reg.numpy()
        )

        all_reg_pred.append(
            regression.cpu().numpy()
        )

        # Horizontal

        all_h_true.extend(
            target_h.numpy().tolist()
        )

        all_h_pred.extend(
            torch.argmax(
                horizontal,
                dim=1
            ).cpu().numpy().tolist()
        )

        # Vertical

        all_v_true.extend(
            target_v.numpy().tolist()
        )

        all_v_pred.extend(
            torch.argmax(
                vertical,
                dim=1
            ).cpu().numpy().tolist()
        )

        # Direction

        all_d_true.extend(
            target_d.numpy().tolist()
        )

        all_d_pred.extend(
            torch.argmax(
                direction,
                dim=1
            ).cpu().numpy().tolist()
        )

        # Geometry

        all_geom_true.append(
            target_geometry.numpy()
        )

        all_geom_pred.append(
            geometry.cpu().numpy()
        )


# ============================================================
# Convert arrays
# ============================================================

reg_true = np.concatenate(
    all_reg_true,
    axis=0
)

reg_pred = np.concatenate(
    all_reg_pred,
    axis=0
)

geom_true = np.concatenate(
    all_geom_true,
    axis=0
)

geom_pred = np.concatenate(
    all_geom_pred,
    axis=0
)


# ============================================================
# Regression metrics
# ============================================================

yaw_error = np.abs(
    reg_true[:, 0] - reg_pred[:, 0]
)

pitch_error = np.abs(
    reg_true[:, 1] - reg_pred[:, 1]
)

overall_error = np.mean(
    np.abs(reg_true - reg_pred),
    axis=1
)

# Angular error using yaw/pitch components
angular_error = np.sqrt(
    (
        reg_true[:, 0]
        - reg_pred[:, 0]
    ) ** 2
    +
    (
        reg_true[:, 1]
        - reg_pred[:, 1]
    ) ** 2
)


# ============================================================
# Print regression
# ============================================================

print()
print("=" * 70)
print("REGRESSION")
print("=" * 70)

print(
    f"Yaw MAE           : {yaw_error.mean():.3f}°"
)

print(
    f"Pitch MAE         : {pitch_error.mean():.3f}°"
)

print(
    f"Overall MAE       : {overall_error.mean():.3f}°"
)

print(
    f"Mean angular error: {angular_error.mean():.3f}°"
)

print(
    f"<5°               : "
    f"{np.mean(angular_error < 5) * 100:.2f}%"
)

print(
    f"<10°              : "
    f"{np.mean(angular_error < 10) * 100:.2f}%"
)


# ============================================================
# Classification helper
# ============================================================

def print_classification(
    name,
    y_true,
    y_pred,
    labels=None
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    ) * 100

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    ) * 100

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    ) * 100

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    ) * 100

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall   : {recall:.2f}%")
    print(f"F1       : {f1:.2f}%")

    if labels is not None:

        print()

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            digits=3
        )

        print(report)
# ============================================================
# Classification metrics
# ============================================================

print_classification(
    "HORIZONTAL",
    all_h_true,
    all_h_pred,
    ["LEFT", "CENTER", "RIGHT"]
)

print_classification(
    "VERTICAL",
    all_v_true,
    all_v_pred,
    ["UP", "CENTER", "DOWN"]
)

print_classification(
    "9-CLASS DIRECTION",
    all_d_true,
    all_d_pred,
    DIRECTION_NAMES
)


# ============================================================
# Confusion matrix
# ============================================================

print()
print("=" * 70)
print("9-CLASS CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    all_d_true,
    all_d_pred,
    labels=list(range(9))
)

print(cm)


# ============================================================
# Geometry metrics
# ============================================================

geometry_error = np.abs(
    geom_true - geom_pred
)

geometry_mae = geometry_error.mean(
    axis=0
)

print()
print("=" * 70)
print("GEOMETRY")
print("=" * 70)

print(
    f"Left pupil X MAE  : "
    f"{geometry_mae[0]:.5f}"
)

print(
    f"Left pupil Y MAE  : "
    f"{geometry_mae[1]:.5f}"
)

print(
    f"Right pupil X MAE : "
    f"{geometry_mae[2]:.5f}"
)

print(
    f"Right pupil Y MAE : "
    f"{geometry_mae[3]:.5f}"
)

print(
    f"Overall geometry MAE: "
    f"{geometry_error.mean():.5f}"
)


# ============================================================
# Error samples
# ============================================================

print()
print("=" * 70)
print("MISCLASSIFIED DIRECTION SAMPLES")
print("=" * 70)

test_df = pd.read_csv(TEST_CSV)

wrong_indices = [
    i
    for i, (t, p)
    in enumerate(zip(all_d_true, all_d_pred))
    if t != p
]

print(
    f"Wrong predictions: "
    f"{len(wrong_indices)} / {len(all_d_true)}"
)

for i in wrong_indices:

    row = test_df.iloc[i]

    print(
        f"{row['sample_id']} "
        f"{row['hp_folder']} : "
        f"{DIRECTION_NAMES[all_d_true[i]]} "
        f"-> "
        f"{DIRECTION_NAMES[all_d_pred[i]]}"
    )


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("V5 EVALUATION COMPLETE")
print("=" * 70)
