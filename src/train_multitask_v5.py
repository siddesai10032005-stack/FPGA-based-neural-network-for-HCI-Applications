import os
import random
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# U2EYES MULTI-TASK GAZE V5
#
# V5 = V3 + auxiliary learned pupil-geometry prediction
#
# IMPORTANT:
# Ground-truth pupil coordinates are NOT model inputs.
# They are ONLY auxiliary training targets.
#
# At inference, the model requires only:
#   left eye image
#   right eye image
#
# Outputs:
#   regression  -> yaw, pitch
#   horizontal  -> LEFT/CENTER/RIGHT
#   vertical    -> UP/CENTER/DOWN
#   direction   -> 9 gaze classes
#   geometry    -> normalized pupil x/y for both eyes
# ============================================================


# ============================================================
# CONFIG
# ============================================================

ROOT = "data/processed"

TRAIN_CSV = os.path.join(ROOT, "train.csv")
VAL_CSV = os.path.join(ROOT, "val.csv")

LEFT_DIR = os.path.join(ROOT, "eyes", "left")
RIGHT_DIR = os.path.join(ROOT, "eyes", "right")

# IMPORTANT:
# This is a NEW checkpoint.
MODEL_PATH = "models/best_multitask_v5.pth"

BATCH_SIZE = 32
EPOCHS = 60

LR = 5e-4
WEIGHT_DECAY = 1e-4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LABELS
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


# ============================================================
# DATASET
# ============================================================

class GazeDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.df)

    def load_image(self, path):

        img = Image.open(path).convert("RGB")

        # Safety check
        if img.size != (128, 64):
            img = img.resize((128, 64))

        img = np.asarray(
            img,
            dtype=np.float32
        ) / 255.0

        img = torch.from_numpy(img)

        return img.permute(2, 0, 1)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ----------------------------------------------------
        # Eye images
        # ----------------------------------------------------

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
        # Continuous gaze target
        # ----------------------------------------------------

        yaw = float(row["relative_yaw"])
        pitch = float(row["relative_pitch"])

        target_reg = torch.tensor(
            [yaw, pitch],
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Horizontal
        # ----------------------------------------------------

        h_name = str(
            row["target_horizontal"]
        ).upper()

        h = HORIZONTAL[h_name]

        # ----------------------------------------------------
        # Vertical
        # ----------------------------------------------------

        v_name = str(
            row["target_vertical"]
        ).upper()

        v = VERTICAL[v_name]

        # ----------------------------------------------------
        # 9-class direction
        # ----------------------------------------------------

        d_name = str(
            row["target_direction"]
        ).upper()

        d = DIRECTION[d_name]

        # ====================================================
        # AUXILIARY PUPIL GEOMETRY TARGET
        #
        # Normalize pupil position relative to the eye crop.
        #
        # [left_x, left_y, right_x, right_y]
        #
        # These values are NOT given to the model.
        # They are only supervision.
        # ====================================================

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

        left_px = (
            float(row["left_pupil_x"])
            - float(row["left_crop_xmin"])
        ) / left_w

        left_py = (
            float(row["left_pupil_y"])
            - float(row["left_crop_ymin"])
        ) / left_h

        right_px = (
            float(row["right_pupil_x"])
            - float(row["right_crop_xmin"])
        ) / right_w

        right_py = (
            float(row["right_pupil_y"])
            - float(row["right_crop_ymin"])
        ) / right_h

        target_geometry = torch.tensor(
            [
                left_px,
                left_py,
                right_px,
                right_py
            ],
            dtype=torch.float32
        )

        return (
            left,
            right,
            target_reg,
            torch.tensor(h, dtype=torch.long),
            torch.tensor(v, dtype=torch.long),
            torch.tensor(d, dtype=torch.long),
            target_geometry
        )


# ============================================================
# EYE CNN
#
# Same V3 backbone.
# ============================================================

class EyeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

    def forward(self, x):

        return self.features(x).flatten(1)


# ============================================================
# V5 MODEL
# ============================================================

class MultiTaskGazeV5(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Shared binocular CNN
        # ----------------------------------------------------

        self.eye_cnn = EyeCNN()

        # 64 left + 64 right = 128
        self.shared = nn.Sequential(

            nn.Linear(
                128,
                96
            ),

            nn.ReLU(),

            nn.Dropout(0.10)
        )

        # ----------------------------------------------------
        # Regression head
        # ----------------------------------------------------

        self.regression_head = nn.Sequential(

            nn.Linear(
                96,
                48
            ),

            nn.ReLU(),

            nn.Linear(
                48,
                2
            )
        )

        # ----------------------------------------------------
        # Horizontal head
        # ----------------------------------------------------

        self.horizontal_head = nn.Sequential(

            nn.Linear(
                96,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                3
            )
        )

        # ----------------------------------------------------
        # Vertical head
        # ----------------------------------------------------

        self.vertical_head = nn.Sequential(

            nn.Linear(
                96,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                3
            )
        )

        # ----------------------------------------------------
        # Direction fusion
        #
        # 96 shared
        # + 3 horizontal
        # + 3 vertical
        # = 102
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                102,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.10)
        )

        # ----------------------------------------------------
        # 9-class direction
        # ----------------------------------------------------

        self.direction_head = nn.Linear(
            64,
            9
        )

        # ====================================================
        # NEW V5 GEOMETRY HEAD
        #
        # Predict:
        #   left pupil x
        #   left pupil y
        #   right pupil x
        #   right pupil y
        #
        # from visual features.
        # ====================================================

        self.geometry_head = nn.Sequential(

            nn.Linear(
                96,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                4
            ),

            nn.Sigmoid()
        )

    def forward(self, left, right):

        # ----------------------------------------------------
        # Extract eye features
        # ----------------------------------------------------

        left_features = self.eye_cnn(left)

        right_features = self.eye_cnn(right)

        # ----------------------------------------------------
        # Binocular fusion
        # ----------------------------------------------------

        features = torch.cat(
            [
                left_features,
                right_features
            ],
            dim=1
        )

        features = self.shared(features)

        # ----------------------------------------------------
        # Regression
        # ----------------------------------------------------

        regression = self.regression_head(
            features
        )

        # ----------------------------------------------------
        # Horizontal
        # ----------------------------------------------------

        horizontal = self.horizontal_head(
            features
        )

        # ----------------------------------------------------
        # Vertical
        # ----------------------------------------------------

        vertical = self.vertical_head(
            features
        )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

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
# SETUP
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("U2EYES MULTI-TASK GAZE V5 TRAINING")
print("=" * 70)

print("Device:", device)
print()


# ============================================================
# DATA
# ============================================================

train_dataset = GazeDataset(
    TRAIN_CSV
)

val_dataset = GazeDataset(
    VAL_CSV
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Val samples  :",
    len(val_dataset)
)

print()


# ============================================================
# MODEL
# ============================================================

model = MultiTaskGazeV5().to(device)

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Trainable parameters:",
    parameters
)

print()


# ============================================================
# LOSSES
# ============================================================

loss_reg = nn.SmoothL1Loss()

loss_h = nn.CrossEntropyLoss()

loss_v = nn.CrossEntropyLoss()

loss_d = nn.CrossEntropyLoss()

loss_geometry = nn.SmoothL1Loss()


# ============================================================
# LOSS WEIGHTS
#
# These intentionally stay close to V3.
#
# Only geometry is new.
# ============================================================

REG_WEIGHT = 1.0

H_WEIGHT = 1.0

V_WEIGHT = 1.0

D_WEIGHT = 2.0

GEOMETRY_WEIGHT = 0.10


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# ============================================================
# BEST CHECKPOINT
# ============================================================

best_score = float("inf")

best_epoch = 0


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    for (
        left,
        right,
        target_reg,
        target_h,
        target_v,
        target_d,
        target_geometry
    ) in train_loader:

        left = left.to(device)

        right = right.to(device)

        target_reg = target_reg.to(device)

        target_h = target_h.to(device)

        target_v = target_v.to(device)

        target_d = target_d.to(device)

        target_geometry = target_geometry.to(device)

        optimizer.zero_grad()

        (
            pred_reg,
            pred_h,
            pred_v,
            pred_d,
            pred_geometry
        ) = model(
            left,
            right
        )

        # ----------------------------------------------------
        # Individual losses
        # ----------------------------------------------------

        l_reg = loss_reg(
            pred_reg,
            target_reg
        )

        l_h = loss_h(
            pred_h,
            target_h
        )

        l_v = loss_v(
            pred_v,
            target_v
        )

        l_d = loss_d(
            pred_d,
            target_d
        )

        l_geometry = loss_geometry(
            pred_geometry,
            target_geometry
        )

        # ----------------------------------------------------
        # Total V5 loss
        # ----------------------------------------------------

        total = (

            REG_WEIGHT * l_reg

            + H_WEIGHT * l_h

            + V_WEIGHT * l_v

            + D_WEIGHT * l_d

            + GEOMETRY_WEIGHT * l_geometry
        )

        total.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        running_loss += total.item()

    running_loss /= len(
        train_loader
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    mae = 0.0

    correct_h = 0

    correct_v = 0

    correct_d = 0

    geometry_error = 0.0

    total_samples = 0

    batches = 0

    with torch.no_grad():

        for (
            left,
            right,
            target_reg,
            target_h,
            target_v,
            target_d,
            target_geometry
        ) in val_loader:

            left = left.to(device)

            right = right.to(device)

            target_reg = target_reg.to(device)

            target_h = target_h.to(device)

            target_v = target_v.to(device)

            target_d = target_d.to(device)

            target_geometry = target_geometry.to(device)

            (
                pred_reg,
                pred_h,
                pred_v,
                pred_d,
                pred_geometry
            ) = model(
                left,
                right
            )

            # ------------------------------------------------
            # Regression MAE
            # ------------------------------------------------

            mae += torch.abs(
                pred_reg - target_reg
            ).mean().item()

            # ------------------------------------------------
            # Geometry MAE
            # ------------------------------------------------

            geometry_error += torch.abs(
                pred_geometry
                - target_geometry
            ).mean().item()

            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            out_h = torch.argmax(
                pred_h,
                dim=1
            )

            out_v = torch.argmax(
                pred_v,
                dim=1
            )

            out_d = torch.argmax(
                pred_d,
                dim=1
            )

            correct_h += (
                out_h == target_h
            ).sum().item()

            correct_v += (
                out_v == target_v
            ).sum().item()

            correct_d += (
                out_d == target_d
            ).sum().item()

            total_samples += target_d.size(0)

            batches += 1

    mae /= batches

    geometry_error /= batches

    acc_h = (
        100.0
        * correct_h
        / total_samples
    )

    acc_v = (
        100.0
        * correct_v
        / total_samples
    )

    acc_d = (
        100.0
        * correct_d
        / total_samples
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(mae)

    current_lr = optimizer.param_groups[0]["lr"]

    # ========================================================
    # CHECKPOINT SCORE
    #
    # Direction is the primary objective.
    # Vertical is next.
    # Regression is secondary.
    #
    # Same basic selection philosophy as our strong V3.
    # ========================================================

    score = (
        2.0 * (1.0 - acc_d / 100.0)

        + 1.5 * (1.0 - acc_v / 100.0)

        + mae / 10.0
    )

    is_best = score < best_score

    if is_best:

        best_score = score

        best_epoch = epoch + 1

        os.makedirs(
            "models",
            exist_ok=True
        )

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "val_mae":
                    mae,

                "horizontal_accuracy":
                    acc_h,

                "vertical_accuracy":
                    acc_v,

                "direction_accuracy":
                    acc_d,

                "geometry_mae":
                    geometry_error,

                "selection_score":
                    score
            },
            MODEL_PATH
        )

    marker = "* " if is_best else ""

    print(
        f"Epoch {epoch + 1:02d} "
        f"Loss {running_loss:.4f} "
        f"MAE {mae:.3f} "
        f"H {acc_h:.1f}% "
        f"V {acc_v:.1f}% "
        f"DIR {acc_d:.1f}% "
        f"GEOM {geometry_error:.4f} "
        f"LR {current_lr:.6f} "
        f"{marker}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)
print("V5 TRAINING COMPLETE")
print("=" * 70)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation score:",
    f"{best_score:.6f}"
)

print(
    "Saved checkpoint:",
    MODEL_PATH
)

print()

print(
    "V3 and V4 checkpoints were not modified."
)

print(
    "Test set was not used during training."
)

print("=" * 70)
