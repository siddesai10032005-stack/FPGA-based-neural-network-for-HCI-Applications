import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from pytorch_nndct.apis import torch_quantizer


# ============================================================
# PATHS
# ============================================================

ROOT = "/workspace/u2eyes_fpga"

CHECKPOINT = os.path.join(
    ROOT, "models/best_multitask_v5.pth"
)

TRAIN_CSV = os.path.join(
    ROOT, "data/processed/train.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT, "models/v5_int8_final"
)

DEVICE = torch.device("cpu")

CALIBRATION_SAMPLES = 300


# ============================================================
# V5 MODEL
# EXACT CHECKPOINT ARCHITECTURE
# ============================================================


def _remap_eye_cnn_state_dict(state_dict):
    """Duplicate legacy 'eye_cnn.*' checkpoint keys into
    'eye_cnn_left.*' and 'eye_cnn_right.*' so old checkpoints
    (saved before the branch split) still load correctly."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("eye_cnn."):
            suffix = k[len("eye_cnn."):]
            new_sd["eye_cnn_left." + suffix] = v
            new_sd["eye_cnn_right." + suffix] = v.clone()
        else:
            new_sd[k] = v
    return new_sd

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

        # Two structurally separate branches (required so the Vitis AI
        # compiler doesn't treat this as a shared/reused module, which was
        # corrupting the XIR graph). Weights are tied below so both
        # branches behave identically to a single shared module.
        self.eye_cnn_left = EyeCNN()
        self.eye_cnn_right = EyeCNN()

        for p_left, p_right in zip(
            self.eye_cnn_left.parameters(),
            self.eye_cnn_right.parameters()
        ):
            p_right.data = p_left.data

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

        # Single call site for eye_cnn: stack both eyes along the batch
        # dimension instead of calling the shared module twice. This is
        # mathematically identical (same weights applied to each eye) but
        # avoids the tracer failing to capture output-tensor shape info
        # for the second call site, which was corrupting the XIR graph
        # during xmodel export.
        left_feat = self.eye_cnn_left(left)
        right_feat = self.eye_cnn_right(right)

        x = torch.cat(
            [left_feat, right_feat],
            dim=1
        )

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
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(path_value):

    """
    CSV stores filenames such as:
        001_01.png

    Do NOT assume they are relative to ROOT.

    Search the actual processed crop directories.
    """

    filename = os.path.basename(str(path_value))

    # If CSV already contains a valid absolute path
    if os.path.isabs(str(path_value)):
        if os.path.exists(str(path_value)):
            return str(path_value)

    # If the path stored in CSV is already valid relative to ROOT
    candidate = os.path.join(
        ROOT,
        str(path_value)
    )

    if os.path.exists(candidate):
        return candidate

    # Known project locations
    search_roots = [
        os.path.join(ROOT, "data"),
        os.path.join(ROOT, "data", "processed"),
        os.path.join(ROOT, "data", "processed", "left"),
        os.path.join(ROOT, "data", "processed", "right"),
        os.path.join(ROOT, "data", "processed", "crops"),
        os.path.join(ROOT, "data", "processed", "eye_crops"),
    ]

    # Direct search first
    for search_root in search_roots:

        candidate = os.path.join(
            search_root,
            filename
        )

        if os.path.exists(candidate):
            return candidate

    # Recursive search as final fallback
    matches = glob.glob(
        os.path.join(
            ROOT,
            "**",
            filename
        ),
        recursive=True
    )

    if len(matches) > 0:
        return matches[0]

    raise FileNotFoundError(
        "\nCould not locate image from CSV.\n"
        f"CSV value: {path_value}\n"
        f"Filename:  {filename}\n"
        f"Searched under: {ROOT}\n"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path_value):

    path = resolve_image_path(path_value)

    img = Image.open(path).convert("RGB")

    img = img.resize(
        (128, 64)
    )

    arr = np.asarray(
        img,
        dtype=np.float32
    ) / 255.0

    arr = np.transpose(
        arr,
        (2, 0, 1)
    )

    return torch.from_numpy(arr)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("U2EYES V5 — FINAL INT8 CALIBRATION + TEST + XMODEL")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\n[1/5] Loading V5 checkpoint...")

model = MultiTaskV5()

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]

else:
    state_dict = checkpoint

model.load_state_dict(_remap_eye_cnn_state_dict(
    state_dict,
    strict=True
))

model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# 2. LOAD CALIBRATION DATA
# ============================================================

print("\n[2/5] Loading calibration samples...")

df = pd.read_csv(
    TRAIN_CSV
)

df = df.head(
    CALIBRATION_SAMPLES
)

print(
    "Calibration samples:",
    len(df)
)

print("\nResolving first image paths...")

# Resolve first sample BEFORE creating quantizer.
# This prevents another long run from failing later.

first_left = resolve_image_path(
    df.iloc[0]["left_image"]
)

first_right = resolve_image_path(
    df.iloc[0]["right_image"]
)

print("First left :", first_left)
print("First right:", first_right)


calibration_pairs = []

for _, row in df.iterrows():

    left = load_image(
        row["left_image"]
    )

    right = load_image(
        row["right_image"]
    )

    calibration_pairs.append(
        (
            left.unsqueeze(0),
            right.unsqueeze(0)
        )
    )


print(
    "\nLoaded:",
    len(calibration_pairs),
    "calibration pairs"
)


# ============================================================
# 3. VITIS AI CALIBRATION
# ============================================================

print(
    "\n[3/5] Starting Vitis AI INT8 calibration..."
)

dummy_left = calibration_pairs[0][0]
dummy_right = calibration_pairs[0][1]

quantizer = torch_quantizer(
    "calib",
    model,
    (dummy_left, dummy_right),
    device=DEVICE,
    output_dir=OUTPUT_DIR,
    bitwidth=8
)


def calibration_run():

    with torch.no_grad():

        for i, (left, right) in enumerate(
            calibration_pairs,
            start=1
        ):

            quantizer.quant_model(
                left,
                right
            )

            if (
                i == 1
                or i % 25 == 0
                or i == len(calibration_pairs)
            ):

                print(
                    f"Calibration: "
                    f"{i}/{len(calibration_pairs)}"
                )


quantizer.quantize(
    calibration_run,
    ()
)

print(
    "\nINT8 calibration completed."
)


# ============================================================
# 4. QUANTIZED TEST
# ============================================================

print(
    "\n[4/5] Testing quantized model..."
)


def test_run():

    with torch.no_grad():

        for i, (left, right) in enumerate(
            calibration_pairs[:20],
            start=1
        ):

            quantizer.quant_model(
                left,
                right
            )

    print(
        "Quantized forward passes: 20"
    )


quantizer.test(
    test_run,
    ()
)

print(
    "INT8 test completed."
)


# ============================================================
# 5. EXPORT XMODEL
# ============================================================

print(
    "\n[5/5] Exporting XModel..."
)

XMODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "xmodel"
)

os.makedirs(
    XMODEL_DIR,
    exist_ok=True
)


# One clean batch-1 inference before export.
with torch.no_grad():

    quantizer.quant_model(
        dummy_left,
        dummy_right
    )


quantizer.export_xmodel(
    output_dir=XMODEL_DIR,
    deploy_check=False,
    dynamic_batch=False
)


# ============================================================
# SEARCH FOR XMODEL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SEARCHING FOR XMODEL"
)

print(
    "=" * 70
)

found = []

for root, dirs, files in os.walk(
    XMODEL_DIR
):

    for filename in files:

        if filename.endswith(
            ".xmodel"
        ):

            found.append(
                os.path.join(
                    root,
                    filename
                )
            )


if not found:

    print(
        "\nWARNING: No .xmodel file found."
    )

else:

    print(
        "\nSUCCESS — XMODEL CREATED!"
    )

    for path in found:

        size = os.path.getsize(
            path
        )

        print(
            "\nXMODEL:",
            path
        )

        print(
            "SIZE:",
            size,
            "bytes"
        )


print(
    "\n" + "=" * 70
)

print(
    "V5 INT8 PIPELINE FINISHED"
)

print(
    "=" * 70
)
