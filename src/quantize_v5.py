import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_nndct.apis import torch_quantizer


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = "/workspace/u2eyes_fpga"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "best_multitask_v5.pth"
)

TRAIN_CSV = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "train.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "v5_int8"
)

IMG_W = 128
IMG_H = 64

BATCH_SIZE = 1

# Number of samples used for INT8 calibration
CALIBRATION_SAMPLES = 300

DEVICE = torch.device("cpu")


# ============================================================
# V5 MODEL
# ============================================================

class EyeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class MultiTaskV5(nn.Module):

    def __init__(self):
        super().__init__()

        self.eye_cnn = EyeCNN()

        self.shared = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.regression_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 2)
        )

        self.horizontal_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        self.vertical_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        self.geometry_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(102, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.direction_head = nn.Linear(64, 9)

    def forward(self, left_eye, right_eye):

        left_features = self.eye_cnn(left_eye)
        right_features = self.eye_cnn(right_eye)

        combined = torch.cat(
            [left_features, right_features],
            dim=1
        )

        shared = self.shared(combined)

        regression = self.regression_head(shared)
        horizontal = self.horizontal_head(shared)
        vertical = self.vertical_head(shared)
        geometry = self.geometry_head(shared)

        fusion_input = torch.cat(
            [
                shared,
                horizontal,
                vertical
            ],
            dim=1
        )

        fusion_features = self.fusion(
            fusion_input
        )

        direction = self.direction_head(
            fusion_features
        )

        return (
            regression,
            horizontal,
            vertical,
            direction,
            geometry
        )


# ============================================================
# DATASET
# ============================================================

class EyeDataset(Dataset):

    def __init__(self, csv_path, limit=None):

        self.df = pd.read_csv(csv_path)

        if limit is not None:
            self.df = self.df.iloc[:limit].copy()

        print(
            f"Dataset samples: {len(self.df)}"
        )

    def __len__(self):
        return len(self.df)

    def load_eye(self, path):

        image = Image.open(path).convert("RGB")

        image = image.resize(
            (IMG_W, IMG_H)
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        return torch.from_numpy(image)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        left_path = os.path.join(
            PROJECT_DIR,
            "data",
            "processed",
            "eyes",
            "left",
            os.path.basename(
                row["left_image"]
            )
        )

        right_path = os.path.join(
            PROJECT_DIR,
            "data",
            "processed",
            "eyes",
            "right",
            os.path.basename(
                row["right_image"]
            )
        )

        left = self.load_eye(
            left_path
        )

        right = self.load_eye(
            right_path
        )

        return left, right


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("============================================")
print("V5 INT8 QUANTIZATION")
print("============================================")
print()

print("Loading V5 model...")

model = MultiTaskV5().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("V5 checkpoint loaded.")
print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# ============================================================
# CALIBRATION DATA
# ============================================================

print()
print(
    f"Loading {CALIBRATION_SAMPLES} "
    "calibration samples..."
)

dataset = EyeDataset(
    TRAIN_CSV,
    limit=CALIBRATION_SAMPLES
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Calibration data ready.")


# ============================================================
# EXAMPLE INPUT
# ============================================================

example_left, example_right = next(
    iter(loader)
)

example_left = example_left.to(
    DEVICE
)

example_right = example_right.to(
    DEVICE
)

print()
print(
    "Left input shape:",
    example_left.shape
)

print(
    "Right input shape:",
    example_right.shape
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# QUANTIZER
# ============================================================

print()
print("Creating Vitis AI INT8 quantizer...")
print()

quantizer = torch_quantizer(
    "calib",
    model,
    (example_left, example_right),
    device=DEVICE,
    output_dir=OUTPUT_DIR,
    bitwidth=8
)

quant_model = quantizer.quant_model

quant_model.eval()


# ============================================================
# CALIBRATION
# ============================================================

print()
print("============================================")
print("STARTING INT8 CALIBRATION")
print("============================================")
print()

with torch.no_grad():

    for index, (left, right) in enumerate(
        loader
    ):

        left = left.to(DEVICE)
        right = right.to(DEVICE)

        quant_model(
            left,
            right
        )

        if (index + 1) % 25 == 0:

            print(
                f"Calibration samples: "
                f"{index + 1}/{CALIBRATION_SAMPLES}"
            )


# ============================================================
# EXPORT CALIBRATION RESULTS
# ============================================================

print()
print("Exporting INT8 calibration model...")

quantizer.export_quant_config()

print()
print("============================================")
print("INT8 CALIBRATION COMPLETE")
print("============================================")
print()
print(
    "Output directory:",
    OUTPUT_DIR
)
print()
print("Files:")
print(
    os.listdir(OUTPUT_DIR)
)
print()
