import os
import sys
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from pytorch_nndct.apis import torch_quantizer


# ============================================================
# CONFIG
# ============================================================

ROOT = "/workspace/u2eyes_fpga"

CHECKPOINT = os.path.join(ROOT, "models/best_multitask_v5.pth")
QUANT_DIR = os.path.join(ROOT, "models/v5_int8")
QUANT_CONFIG = os.path.join(QUANT_DIR, "quant_info.json")

TEST_CSV = os.path.join(ROOT, "data/processed/test.csv")

DEVICE = torch.device("cpu")

print("=" * 70)
print("U2EYES V5 — VITIS AI INT8 VERIFICATION + XMODEL EXPORT")
print("=" * 70)

print("\nCheckpoint :", CHECKPOINT)
print("Quant dir  :", QUANT_DIR)
print("Quant info :", QUANT_CONFIG)
print("Test CSV   :", TEST_CSV)


# ============================================================
# V5 MODEL — EXACT ARCHITECTURE
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
        x = self.features(x)
        return torch.flatten(x, 1)


class MultiTaskV5(nn.Module):
    def __init__(self):
        super().__init__()

        # IMPORTANT:
        # V5 uses ONE shared EyeCNN for both left and right eyes.
        self.eye_cnn = EyeCNN()

        self.shared = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.10)
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
            nn.Dropout(0.10)
        )

        self.direction_head = nn.Linear(64, 9)

    def forward(self, left, right):

        # SAME backbone processes both eyes
        left_feat = self.eye_cnn(left)
        right_feat = self.eye_cnn(right)

        x = torch.cat([left_feat, right_feat], dim=1)

        shared = self.shared(x)

        regression = self.regression_head(shared)
        horizontal = self.horizontal_head(shared)
        vertical = self.vertical_head(shared)
        geometry = self.geometry_head(shared)

        fusion_input = torch.cat(
            [shared, horizontal, vertical],
            dim=1
        )

        fused = self.fusion(fusion_input)

        direction = self.direction_head(fused)

        return (
            regression,
            horizontal,
            vertical,
            geometry,
            direction
        )


# ============================================================
# LOAD FP32 MODEL
# ============================================================

print("\n[1/6] Loading V5 checkpoint...")

model = MultiTaskV5()

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

model.eval()
model.to(DEVICE)

print("FP32 checkpoint loaded successfully.")


# ============================================================
# CHECK INPUT
# ============================================================

dummy_left = torch.randn(1, 3, 64, 128)
dummy_right = torch.randn(1, 3, 64, 128)

print("\n[2/6] Checking FP32 model...")

with torch.no_grad():
    fp32_output = model(dummy_left, dummy_right)

print("FP32 forward successful.")

for i, out in enumerate(fp32_output):
    print(
        f"  Output {i}: "
        f"shape={tuple(out.shape)} "
        f"dtype={out.dtype}"
    )


# ============================================================
# CREATE VITIS AI TEST QUANTIZER
# ============================================================

print("\n[3/6] Loading existing INT8 calibration configuration...")

if not os.path.exists(QUANT_CONFIG):
    raise FileNotFoundError(
        f"Missing quantization configuration: {QUANT_CONFIG}"
    )

# IMPORTANT:
# quant_info.json is supplied directly so Vitis AI reconstructs
# the calibrated quantization configuration rather than creating
# a fresh calibration.

quantizer = torch_quantizer(
    "test",
    model,
    (dummy_left, dummy_right),
    device=DEVICE,
    output_dir=QUANT_DIR,
    bitwidth=8,
    quant_config_file=QUANT_CONFIG
)

quant_model = quantizer.quant_model

quant_model.eval()

print("Vitis AI quantized model created successfully.")


# ============================================================
# INT8 DUMMY FORWARD
# ============================================================

print("\n[4/6] Running INT8 forward pass...")

with torch.no_grad():
    int8_output = quant_model(dummy_left, dummy_right)

print("INT8 forward SUCCESS.")

if isinstance(int8_output, (tuple, list)):
    print("Number of outputs:", len(int8_output))

    for i, out in enumerate(int8_output):
        if torch.is_tensor(out):
            print(
                f"  Output {i}: "
                f"shape={tuple(out.shape)} "
                f"dtype={out.dtype}"
            )
        else:
            print(f"  Output {i}: {type(out)}")
else:
    print("Output type:", type(int8_output))


# ============================================================
# EXPORT XMODEL
# ============================================================

print("\n[5/6] Exporting XModel...")

EXPORT_DIR = os.path.join(QUANT_DIR, "xmodel")
os.makedirs(EXPORT_DIR, exist_ok=True)

quantizer.export_xmodel(
    output_dir=EXPORT_DIR,
    deploy_check=False,
    dynamic_batch=False
)

print("\nXMODEL EXPORT COMPLETED.")


# ============================================================
# SHOW RESULT
# ============================================================

print("\n[6/6] Searching for generated XModel...")

for root, dirs, files in os.walk(EXPORT_DIR):
    for f in files:
        if f.endswith(".xmodel"):
            path = os.path.join(root, f)
            size = os.path.getsize(path)

            print("\n" + "=" * 70)
            print("SUCCESS — XMODEL CREATED")
            print("=" * 70)
            print("File :", path)
            print("Size :", size, "bytes")
            print("=" * 70)

print("\nDONE.")
