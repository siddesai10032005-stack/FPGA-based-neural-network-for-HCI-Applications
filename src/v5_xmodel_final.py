import os
import glob
import shutil
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

# Use a fresh directory so no stale XModel artifacts interfere.
OUTPUT_DIR = os.path.join(
    ROOT,
    "models",
    "v5_xmodel_final"
)

DEVICE = torch.device("cpu")

CALIBRATION_SAMPLES = 300

# The deployment representation uses mathematically equivalent
# pooling and Linear -> 1x1 Conv transformations. Small floating
# point differences are expected.
EQUIVALENCE_TOLERANCE = 1e-3


# ============================================================
# ORIGINAL V5 EYE CNN
# ============================================================

class OriginalEyeCNN(nn.Module):

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

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

    def forward(self, x):

        x = self.features(x)

        return torch.flatten(
            x,
            1
        )


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

    def forward(
        self,
        left,
        right
    ):

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
# DEPLOYMENT EYE CNN
#
# Original:
#   AdaptiveAvgPool2d(1,1)
#
# For fixed V5 input [B,3,64,128]:
#
#   after pool 1 -> H=32 W=64
#   after pool 2 -> H=16 W=32
#
# final feature map:
#   [B,64,16,32]
#
# Therefore:
#   AvgPool2d((16,32))
#
# is mathematically equivalent to global average pooling.
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

            nn.AvgPool2d(
                kernel_size=(16, 32),
                stride=(16, 32)
            )
        )

    def forward(self, x):

        return self.features(x)


# ============================================================
# DEPLOYMENT V5
#
# IMPORTANT:
# Everything stays 4-D.
#
# Linear -> mathematically equivalent 1x1 Conv2d.
# ============================================================

class DeployV5(nn.Module):

    def __init__(self):
        super().__init__()

        self.eye_cnn = DeployEyeCNN()

        # 128 -> 96
        self.shared = nn.Sequential(
            nn.Conv2d(
                128,
                96,
                kernel_size=1
            ),
            nn.ReLU()
        )

        # 96 -> 48 -> 2
        self.regression_head = nn.Sequential(
            nn.Conv2d(
                96,
                48,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                48,
                2,
                kernel_size=1
            )
        )

        # 96 -> 32 -> 3
        self.horizontal_head = nn.Sequential(
            nn.Conv2d(
                96,
                32,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                3,
                kernel_size=1
            )
        )

        # 96 -> 32 -> 3
        self.vertical_head = nn.Sequential(
            nn.Conv2d(
                96,
                32,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                3,
                kernel_size=1
            )
        )

        # 96 -> 32 -> 4
        self.geometry_head = nn.Sequential(
            nn.Conv2d(
                96,
                32,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                4,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

        # 96 + 3 + 3 = 102 -> 64
        self.fusion = nn.Sequential(
            nn.Conv2d(
                102,
                64,
                kernel_size=1
            ),
            nn.ReLU()
        )

        # 64 -> 9
        self.direction_head = nn.Conv2d(
            64,
            9,
            kernel_size=1
        )

    def forward(
        self,
        left,
        right
    ):

        # [B,64,1,1]
        left_feat = self.eye_cnn(left)

        # [B,64,1,1]
        right_feat = self.eye_cnn(right)

        # [B,128,1,1]
        x = torch.cat(
            [
                left_feat,
                right_feat
            ],
            dim=1
        )

        # [B,96,1,1]
        shared = self.shared(x)

        # [B,2,1,1]
        regression = self.regression_head(
            shared
        )

        # [B,3,1,1]
        horizontal = self.horizontal_head(
            shared
        )

        # [B,3,1,1]
        vertical = self.vertical_head(
            shared
        )

        # [B,4,1,1]
        geometry = self.geometry_head(
            shared
        )

        # [B,102,1,1]
        fusion_input = torch.cat(
            [
                shared,
                horizontal,
                vertical
            ],
            dim=1
        )

        # [B,64,1,1]
        fused = self.fusion(
            fusion_input
        )

        # [B,9,1,1]
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
# WEIGHT TRANSFER
#
# Linear:
#   [out, in]
#
# 1x1 Conv:
#   [out, in, 1, 1]
#
# Same exact learned weights.
# ============================================================

def copy_linear_to_conv(
    linear,
    conv
):

    with torch.no_grad():

        conv.weight.copy_(
            linear.weight.reshape(
                linear.out_features,
                linear.in_features,
                1,
                1
            )
        )

        conv.bias.copy_(
            linear.bias
        )


def transfer_weights(
    original,
    deploy
):

    # --------------------------------------------------------
    # Eye CNN
    # --------------------------------------------------------

    eye_pairs = [
        (0, 0),
        (1, 1),
        (4, 4),
        (5, 5),
        (8, 8),
        (9, 9),
        (11, 11),
        (12, 12)
    ]

    for src_idx, dst_idx in eye_pairs:

        deploy.eye_cnn.features[
            dst_idx
        ].load_state_dict(
            original.eye_cnn.features[
                src_idx
            ].state_dict()
        )

    # --------------------------------------------------------
    # Shared
    # --------------------------------------------------------

    copy_linear_to_conv(
        original.shared[0],
        deploy.shared[0]
    )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    copy_linear_to_conv(
        original.regression_head[0],
        deploy.regression_head[0]
    )

    copy_linear_to_conv(
        original.regression_head[2],
        deploy.regression_head[2]
    )

    # --------------------------------------------------------
    # Horizontal
    # --------------------------------------------------------

    copy_linear_to_conv(
        original.horizontal_head[0],
        deploy.horizontal_head[0]
    )

    copy_linear_to_conv(
        original.horizontal_head[2],
        deploy.horizontal_head[2]
    )

    # --------------------------------------------------------
    # Vertical
    # --------------------------------------------------------

    copy_linear_to_conv(
        original.vertical_head[0],
        deploy.vertical_head[0]
    )

    copy_linear_to_conv(
        original.vertical_head[2],
        deploy.vertical_head[2]
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    copy_linear_to_conv(
        original.geometry_head[0],
        deploy.geometry_head[0]
    )

    copy_linear_to_conv(
        original.geometry_head[2],
        deploy.geometry_head[2]
    )

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    copy_linear_to_conv(
        original.fusion[0],
        deploy.fusion[0]
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    copy_linear_to_conv(
        original.direction_head,
        deploy.direction_head
    )


# ============================================================
# IMAGE PATH RESOLUTION
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

    # Common processed directories
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
        "\n"
        "IMAGE NOT FOUND\n"
        f"CSV value : {original}\n"
        f"Filename  : {filename}\n"
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
# LOAD CHECKPOINT
# ============================================================

print(
    "=" * 70
)

print(
    "U2EYES V5 — FINAL DEPLOYMENT GRAPH"
)

print(
    "=" * 70
)

print(
    "\n[1/6] Loading V5 checkpoint..."
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


original_model = OriginalV5()

deploy_model = DeployV5()

original_model.load_state_dict(
    state_dict,
    strict=True
)

transfer_weights(
    original_model,
    deploy_model
)

original_model.eval()
deploy_model.eval()

print(
    "Checkpoint loaded."
)

print(
    "Deployment weights transferred."
)


# ============================================================
# SHAPE CHECK
# ============================================================

print(
    "\n[2/6] Checking deployment graph shapes..."
)

torch.manual_seed(
    42
)

dummy_left = torch.randn(
    1,
    3,
    64,
    128
)

dummy_right = torch.randn(
    1,
    3,
    64,
    128
)

with torch.no_grad():

    deploy_output = deploy_model(
        dummy_left,
        dummy_right
    )


names = [
    "regression",
    "horizontal",
    "vertical",
    "geometry",
    "direction"
]

expected_shapes = [
    (1, 2, 1, 1),
    (1, 3, 1, 1),
    (1, 3, 1, 1),
    (1, 4, 1, 1),
    (1, 9, 1, 1)
]


print(
    "Deployment output shapes:"
)

actual_shapes = []

for name, output in zip(
    names,
    deploy_output
):

    shape = tuple(
        output.shape
    )

    actual_shapes.append(
        shape
    )

    print(
        f"  {name:12s}: {shape}"
    )


if actual_shapes != expected_shapes:

    raise RuntimeError(
        "\n"
        "Unexpected deployment output shapes.\n"
        f"Expected: {expected_shapes}\n"
        f"Actual:   {actual_shapes}\n"
    )

print(
    "Shape check PASSED."
)


# ============================================================
# NUMERICAL EQUIVALENCE CHECK
# ============================================================

print(
    "\n[3/6] Comparing deployment graph "
    "against original V5..."
)

with torch.no_grad():

    original_output = original_model(
        dummy_left,
        dummy_right
    )

    deploy_output = deploy_model(
        dummy_left,
        dummy_right
    )


max_diffs = []

for original, deploy in zip(
    original_output,
    deploy_output
):

    # Deployment output is [B,C,1,1].
    deploy_2d = deploy[
        :,
        :,
        0,
        0
    ]

    # Original output is [B,C].
    diff = torch.max(
        torch.abs(
            original - deploy_2d
        )
    ).item()

    max_diffs.append(
        diff
    )


for name, diff in zip(
    names,
    max_diffs
):

    print(
        f"  {name:12s}: "
        f"{diff:.10f}"
    )


overall_diff = max(
    max_diffs
)

print(
    "\nMaximum difference:",
    f"{overall_diff:.10f}"
)

if overall_diff > EQUIVALENCE_TOLERANCE:

    raise RuntimeError(
        "\n"
        "DEPLOYMENT GRAPH DIFFERS TOO MUCH FROM V5.\n"
        f"Maximum difference = {overall_diff:.10f}\n"
        f"Tolerance          = "
        f"{EQUIVALENCE_TOLERANCE:.10f}\n"
    )

print(
    "\nNUMERICAL EQUIVALENCE PASSED."
)

print(
    f"Maximum difference "
    f"{overall_diff:.10f} "
    f"< tolerance "
    f"{EQUIVALENCE_TOLERANCE:.10f}"
)


# ============================================================
# CLASSIFICATION PREDICTION EQUIVALENCE
# ============================================================

print(
    "\nChecking discrete gaze predictions..."
)


# Horizontal
original_h_logits = original_output[1]

deploy_h_logits = deploy_output[1][
    :, :, 0, 0
]

original_h = torch.argmax(
    original_h_logits,
    dim=1
)

deploy_h = torch.argmax(
    deploy_h_logits,
    dim=1
)


# Vertical
original_v_logits = original_output[2]

deploy_v_logits = deploy_output[2][
    :, :, 0, 0
]

original_v = torch.argmax(
    original_v_logits,
    dim=1
)

deploy_v = torch.argmax(
    deploy_v_logits,
    dim=1
)


# Direction
original_dir_logits = original_output[4]

deploy_dir_logits = deploy_output[4][
    :, :, 0, 0
]

original_dir = torch.argmax(
    original_dir_logits,
    dim=1
)

deploy_dir = torch.argmax(
    deploy_dir_logits,
    dim=1
)


h_same = torch.equal(
    original_h,
    deploy_h
)

v_same = torch.equal(
    original_v,
    deploy_v
)

dir_same = torch.equal(
    original_dir,
    deploy_dir
)


print(
    "Horizontal prediction identical:",
    h_same
)

print(
    "Vertical prediction identical:",
    v_same
)

print(
    "Direction prediction identical:",
    dir_same
)


if not (
    h_same
    and v_same
    and dir_same
):

    raise RuntimeError(
        "\n"
        "Deployment graph changes one or more "
        "discrete gaze predictions.\n"
        "Do NOT continue to INT8 export."
    )


print(
    "\nCLASSIFICATION EQUIVALENCE PASSED."
)


# ============================================================
# CALIBRATION DATA
# ============================================================

print(
    "\n[4/6] Loading calibration data..."
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
# FRESH OUTPUT DIRECTORY
# ============================================================

if os.path.exists(
    OUTPUT_DIR
):

    print(
        "\nRemoving previous generated output:"
    )

    print(
        OUTPUT_DIR
    )

    shutil.rmtree(
        OUTPUT_DIR
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# INT8 CALIBRATION
# ============================================================

print(
    "\n[5/6] Starting INT8 calibration..."
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
    "\nTesting quantized model..."
)


def test_run():

    with torch.no_grad():

        for left, right in calibration_pairs[:20]:

            quantizer.quant_model(
                left,
                right
            )

    print(
        "INT8 forward passes: 20"
    )


quantizer.test(
    test_run,
    ()
)

print(
    "INT8 TEST COMPLETE."
)


# ============================================================
# XMODEL DEPLOYMENT
# ============================================================

print(
    "\n[6/6] Exporting XModel..."
)


def deploy_run():

    with torch.no_grad():

        return quantizer.quant_model(
            dummy_left,
            dummy_right
        )


quantizer.deploy(
    deploy_run,
    (),
    fmt="xmodel"
)


# ============================================================
# SEARCH FOR XMODEL
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


if not xmodels:

    print(
        "XMODEL EXPORT FAILED."
    )

    print(
        "No .xmodel file was produced."
    )

    print(
        "=" * 70
    )

    raise RuntimeError(
        "No XModel was produced."
    )


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


print(
    "\n"
    "V5 INT8 -> XMODEL PIPELINE COMPLETE."
)

print(
    "=" * 70
)
