import os
import glob
import json
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

CHECKPOINT = os.path.join(
    ROOT,
    "models",
    "best_multitask_v5.pth"
)

TRAIN_CSV = os.path.join(
    ROOT,
    "data",
    "processed",
    "train.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "models",
    "v5_int8_deploy"
)

DEVICE = torch.device("cpu")

CALIBRATION_SAMPLES = 300


# ============================================================
# ORIGINAL V5 ARCHITECTURE
# Used ONLY to verify checkpoint equivalence.
# ============================================================

class OriginalEyeCNN(nn.Module):

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


# ============================================================
# DEPLOYMENT-SAFE EYE CNN
#
# IMPORTANT:
#
# AdaptiveAvgPool2d(1,1) over a 32x16 feature map is exactly
# equivalent to AvgPool2d(kernel_size=(32,16)).
#
# This removes the problematic adaptive-pool graph.
# ============================================================

class DeployEyeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            # Feature map is exactly:
            # 64 x 32 x 16
            #
            # Average over the complete spatial dimensions.
            nn.AvgPool2d(
                kernel_size=(32, 16)
            )
        )

    def forward(self, x):

        x = self.features(x)

        # After AvgPool:
        # [B, 64, 1, 1]
        #
        # Fixed reshape rather than torch.flatten.
        x = x.reshape(
            x.shape[0],
            64
        )

        return x


# ============================================================
# ORIGINAL V5
# ============================================================

class OriginalV5(nn.Module):

    def __init__(self):
        super().__init__()

        self.eye_cnn = OriginalEyeCNN()

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

        self.direction_head = nn.Linear(
            64,
            9
        )

    def forward(self, left, right):

        left_feat = self.eye_cnn(left)

        right_feat = self.eye_cnn(right)

        x = torch.cat(
            [
                left_feat,
                right_feat
            ],
            dim=1
        )

        shared = self.shared(x)

        regression = self.regression_head(
            shared
        )

        horizontal = self.horizontal_head(
            shared
        )

        vertical = self.vertical_head(
            shared
        )

        geometry = self.geometry_head(
            shared
        )

        fusion_input = torch.cat(
            [
                shared,
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

        return (
            regression,
            horizontal,
            vertical,
            geometry,
            direction
        )


# ============================================================
# DEPLOYMENT-SAFE V5
# SAME WEIGHTS
# SAME OUTPUTS
# DIFFERENT GRAPH REPRESENTATION
# ============================================================

class DeployV5(nn.Module):

    def __init__(self):
        super().__init__()

        self.eye_cnn = DeployEyeCNN()

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

        self.direction_head = nn.Linear(
            64,
            9
        )

    def forward(self, left, right):

        left_feat = self.eye_cnn(left)

        right_feat = self.eye_cnn(right)

        x = torch.cat(
            [
                left_feat,
                right_feat
            ],
            dim=1
        )

        shared = self.shared(x)

        regression = self.regression_head(
            shared
        )

        horizontal = self.horizontal_head(
            shared
        )

        vertical = self.vertical_head(
            shared
        )

        geometry = self.geometry_head(
            shared
        )

        fusion_input = torch.cat(
            [
                shared,
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

        return (
            regression,
            horizontal,
            vertical,
            geometry,
            direction
        )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 70)
print("U2EYES V5 — DEPLOYMENT-SAFE INT8 EXPORT")
print("=" * 70)

print("\n[1/7] Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# LOAD BOTH MODELS
# ============================================================

original_model = OriginalV5()

deploy_model = DeployV5()

original_model.load_state_dict(
    state_dict,
    strict=True
)

deploy_model.load_state_dict(
    state_dict,
    strict=True
)

original_model.eval()
deploy_model.eval()

print(
    "Checkpoint loaded into both models."
)


# ============================================================
# IMAGE RESOLUTION
# ============================================================

def resolve_image_path(
    path_value
):

    original = str(
        path_value
    )

    filename = os.path.basename(
        original
    )

    # Absolute path
    if os.path.isabs(original):

        if os.path.exists(original):
            return original

    # Relative to project
    candidate = os.path.join(
        ROOT,
        original
    )

    if os.path.exists(candidate):
        return candidate

    # Common directories
    search_roots = [

        os.path.join(
            ROOT,
            "data"
        ),

        os.path.join(
            ROOT,
            "data",
            "processed"
        ),

        os.path.join(
            ROOT,
            "data",
            "processed",
            "left"
        ),

        os.path.join(
            ROOT,
            "data",
            "processed",
            "right"
        ),

        os.path.join(
            ROOT,
            "data",
            "processed",
            "crops"
        ),

        os.path.join(
            ROOT,
            "data",
            "processed",
            "eye_crops"
        )
    ]

    for root in search_roots:

        candidate = os.path.join(
            root,
            filename
        )

        if os.path.exists(candidate):
            return candidate

    # Recursive fallback
    matches = glob.glob(
        os.path.join(
            ROOT,
            "**",
            filename
        ),
        recursive=True
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        "\nCould not locate image:\n"
        f"CSV value = {original}\n"
        f"Filename  = {filename}\n"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(
    path_value
):

    path = resolve_image_path(
        path_value
    )

    img = Image.open(
        path
    ).convert(
        "RGB"
    )

    img = img.resize(
        (
            128,
            64
        )
    )

    arr = np.asarray(
        img,
        dtype=np.float32
    )

    arr /= 255.0

    arr = np.transpose(
        arr,
        (
            2,
            0,
            1
        )
    )

    return torch.from_numpy(
        arr
    )


# ============================================================
# VERIFY ARCHITECTURE EQUIVALENCE
# ============================================================

print(
    "\n[2/7] Verifying deployment-safe architecture..."
)

torch.manual_seed(42)

test_left = torch.rand(
    1,
    3,
    64,
    128
)

test_right = torch.rand(
    1,
    3,
    64,
    128
)

with torch.no_grad():

    original_output = original_model(
        test_left,
        test_right
    )

    deploy_output = deploy_model(
        test_left,
        test_right
    )


max_diffs = []

for a, b in zip(
    original_output,
    deploy_output
):

    diff = torch.max(
        torch.abs(
            a - b
        )
    ).item()

    max_diffs.append(
        diff
    )


print(
    "Maximum output differences:"
)

names = [
    "regression",
    "horizontal",
    "vertical",
    "geometry",
    "direction"
]

for name, diff in zip(
    names,
    max_diffs
):

    print(
        f"  {name:12s}: {diff:.10f}"
    )


overall_diff = max(
    max_diffs
)

print(
    f"\nOverall maximum difference: "
    f"{overall_diff:.10f}"
)


if overall_diff > 1e-5:

    raise RuntimeError(
        "Deployment-safe architecture is NOT "
        "numerically equivalent to original V5."
    )


print(
    "\nARCHITECTURE CHECK PASSED."
)

print(
    "The deployment model produces the same "
    "V5 computation."
)


# ============================================================
# LOAD TRAIN CSV
# ============================================================

print(
    "\n[3/7] Loading calibration data..."
)

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
    "Calibration images loaded."
)


# ============================================================
# CREATE FRESH OUTPUT DIRECTORY
# ============================================================

print(
    "\n[4/7] Creating fresh Vitis AI output..."
)

if os.path.exists(
    OUTPUT_DIR
):

    print(
        "Existing output directory found."
    )

    print(
        "Using it as the destination."
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# START VITIS AI QUANTIZER
# ============================================================

print(
    "\n[5/7] Starting INT8 calibration..."
)

dummy_left = calibration_pairs[0][0]

dummy_right = calibration_pairs[0][1]


quantizer = torch_quantizer(

    "calib",

    deploy_model,

    (
        dummy_left,
        dummy_right
    ),

    device=DEVICE,

    output_dir=OUTPUT_DIR,

    bitwidth=8
)


# ============================================================
# CALIBRATION
# ============================================================

def calibration_run():

    with torch.no_grad():

        for i, (
            left,
            right
        ) in enumerate(
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
                or i == len(
                    calibration_pairs
                )
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
    "\nINT8 CALIBRATION COMPLETE."
)


# ============================================================
# INT8 TEST
# ============================================================

print(
    "\n[6/7] Testing INT8 model..."
)


def test_run():

    with torch.no_grad():

        for i, (
            left,
            right
        ) in enumerate(
            calibration_pairs[:20],
            start=1
        ):

            quantizer.quant_model(
                left,
                right
            )

    print(
        "INT8 test forward passes: 20"
    )


quantizer.test(
    test_run,
    ()
)

print(
    "INT8 TEST COMPLETE."
)


# ============================================================
# XMODEL EXPORT
# ============================================================

print(
    "\n[7/7] Exporting XModel..."
)


def deployment_run():

    with torch.no_grad():

        return quantizer.quant_model(
            dummy_left,
            dummy_right
        )


# ------------------------------------------------------------
# DEPLOYMENT
# ------------------------------------------------------------

quantizer.deploy(
    deployment_run,
    (),
    fmt="xmodel"
)


# ============================================================
# FIND XMODEL
# ============================================================

xmodels = []

for root, dirs, files in os.walk(
    OUTPUT_DIR
):

    for filename in files:

        if filename.endswith(
            ".xmodel"
        ):

            xmodels.append(
                os.path.join(
                    root,
                    filename
                )
            )


print(
    "\n"
    + "=" * 70
)

if xmodels:

    print(
        "SUCCESS — XMODEL CREATED"
    )

    print(
        "=" * 70
    )

    for path in xmodels:

        print(
            "\nXMODEL:"
        )

        print(
            path
        )

        print(
            "SIZE:",
            os.path.getsize(path),
            "bytes"
        )

else:

    print(
        "XMODEL EXPORT DID NOT PRODUCE A FILE"
    )

    print(
        "=" * 70
    )

    raise RuntimeError(
        "No .xmodel file was produced."
    )


print(
    "\nV5 INT8 DEPLOYMENT COMPLETE."
)

print(
    "=" * 70
)
