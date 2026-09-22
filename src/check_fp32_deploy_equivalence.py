import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

ROOT = "/workspace/u2eyes_fpga"

CHECKPOINT = os.path.join(
    ROOT,
    "models",
    "best_multitask_v5.pth"
)

TEST_CSV = os.path.join(
    ROOT,
    "data",
    "processed",
    "test.csv"
)

DEVICE = torch.device("cpu")


# ============================================================
# ORIGINAL V5
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

        self.direction_head = nn.Linear(64, 9)

    def forward(self, left, right):

        lf = self.eye_cnn(left)
        rf = self.eye_cnn(right)

        x = torch.cat(
            [lf, rf],
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
            [shared, horizontal, vertical],
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
# ============================================================

class DeployEyeCNN(nn.Module):

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

            # Fixed global average pooling:
            # [B,64,16,32] -> [B,64,1,1]
            nn.AvgPool2d(
                kernel_size=(16, 32),
                stride=(16, 32)
            )
        )

    def forward(self, x):
        return self.features(x)


class DeployV5(nn.Module):

    def __init__(self):
        super().__init__()

        self.eye_cnn = DeployEyeCNN()

        self.shared = nn.Sequential(
            nn.Conv2d(
                128,
                96,
                kernel_size=1
            ),
            nn.ReLU()
        )

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

        self.fusion = nn.Sequential(
            nn.Conv2d(
                102,
                64,
                kernel_size=1
            ),
            nn.ReLU()
        )

        self.direction_head = nn.Conv2d(
            64,
            9,
            kernel_size=1
        )

    def forward(self, left, right):

        lf = self.eye_cnn(left)
        rf = self.eye_cnn(right)

        x = torch.cat(
            [lf, rf],
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
            [shared, horizontal, vertical],
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
# LINEAR -> 1x1 CONV
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


# ============================================================
# LOAD WEIGHTS
# ============================================================

def build_models():

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict):

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


    original = OriginalV5()

    original.load_state_dict(
        state_dict,
        strict=True
    )

    original.eval()


    deploy = DeployV5()


    # --------------------------------------------------------
    # Eye CNN learned layers
    # --------------------------------------------------------

    for idx in [
        0,
        1,
        4,
        5,
        8,
        9,
        11,
        12
    ]:

        deploy.eye_cnn.features[
            idx
        ].load_state_dict(
            original.eye_cnn.features[
                idx
            ].state_dict()
        )


    # --------------------------------------------------------
    # Linear -> 1x1 Conv
    # --------------------------------------------------------

    copy_linear_to_conv(
        original.shared[0],
        deploy.shared[0]
    )

    copy_linear_to_conv(
        original.regression_head[0],
        deploy.regression_head[0]
    )

    copy_linear_to_conv(
        original.regression_head[2],
        deploy.regression_head[2]
    )

    copy_linear_to_conv(
        original.horizontal_head[0],
        deploy.horizontal_head[0]
    )

    copy_linear_to_conv(
        original.horizontal_head[2],
        deploy.horizontal_head[2]
    )

    copy_linear_to_conv(
        original.vertical_head[0],
        deploy.vertical_head[0]
    )

    copy_linear_to_conv(
        original.vertical_head[2],
        deploy.vertical_head[2]
    )

    copy_linear_to_conv(
        original.geometry_head[0],
        deploy.geometry_head[0]
    )

    copy_linear_to_conv(
        original.geometry_head[2],
        deploy.geometry_head[2]
    )

    copy_linear_to_conv(
        original.fusion[0],
        deploy.fusion[0]
    )

    copy_linear_to_conv(
        original.direction_head,
        deploy.direction_head
    )

    deploy.eval()

    return original, deploy


# ============================================================
# IMAGE PATH
# ============================================================

def resolve_image_path(
    value,
    side
):

    original = str(value)

    filename = os.path.basename(
        original
    )

    if os.path.isabs(original):

        if os.path.exists(original):
            return original

    candidate = os.path.join(
        ROOT,
        original
    )

    if os.path.exists(candidate):
        return candidate


    roots = [

        os.path.join(
            ROOT,
            "data",
            "processed",
            side
        ),

        os.path.join(
            ROOT,
            "data",
            "processed",
            "crops",
            side
        ),

        os.path.join(
            ROOT,
            "data",
            "processed",
            "eye_crops",
            side
        ),

        os.path.join(
            ROOT,
            "data",
            "processed"
        )
    ]


    for root in roots:

        candidate = os.path.join(
            root,
            filename
        )

        if os.path.exists(candidate):
            return candidate


    matches = glob.glob(
        os.path.join(
            ROOT,
            "**",
            filename
        ),
        recursive=True
    )


    side_matches = [
        m for m in matches
        if side.lower() in m.lower()
    ]


    if side_matches:
        return side_matches[0]

    if matches:
        return matches[0]


    raise FileNotFoundError(
        f"Image not found: {value}"
    )


# ============================================================
# IMAGE
# ============================================================

def load_image(
    value,
    side
):

    path = resolve_image_path(
        value,
        side
    )

    image = Image.open(
        path
    ).convert(
        "RGB"
    )

    image = image.resize(
        (
            128,
            64
        )
    )

    arr = np.asarray(
        image,
        dtype=np.float32
    )

    arr /= 255.0

    arr = np.transpose(
        arr,
        (2, 0, 1)
    )

    return torch.from_numpy(
        arr
    )


# ============================================================
# LABELS
# ============================================================

H_MAP = {
    "LEFT": 0,
    "CENTER": 1,
    "RIGHT": 2
}

V_MAP = {
    "DOWN": 0,
    "CENTER": 1,
    "UP": 2
}

D_MAP = {
    "DOWN_LEFT": 0,
    "DOWN_CENTER": 1,
    "DOWN_RIGHT": 2,

    "CENTER_LEFT": 3,
    "CENTER_CENTER": 4,
    "CENTER_RIGHT": 5,

    "UP_LEFT": 6,
    "UP_CENTER": 7,
    "UP_RIGHT": 8
}


def label_index(
    value,
    mapping
):

    text = str(
        value
    ).strip().upper().replace(
        " ",
        "_"
    )

    if text in mapping:
        return mapping[text]

    try:
        return int(float(text))
    except Exception:
        raise ValueError(
            f"Unknown label: {value}"
        )


# ============================================================
# METRICS
# ============================================================

def classification_accuracy(
    true,
    pred
):

    true = np.asarray(true)
    pred = np.asarray(pred)

    return float(
        np.mean(
            true == pred
        ) * 100.0
    )


def angle_error(
    true_yaw,
    true_pitch,
    pred_yaw,
    pred_pitch
):

    ty = np.deg2rad(true_yaw)
    tp = np.deg2rad(true_pitch)

    py = np.deg2rad(pred_yaw)
    pp = np.deg2rad(pred_pitch)


    true_vectors = np.stack(
        [
            np.sin(ty) * np.cos(tp),
            np.sin(tp),
            -np.cos(ty) * np.cos(tp)
        ],
        axis=1
    )


    pred_vectors = np.stack(
        [
            np.sin(py) * np.cos(pp),
            np.sin(pp),
            -np.cos(py) * np.cos(pp)
        ],
        axis=1
    )


    dot = np.sum(
        true_vectors * pred_vectors,
        axis=1
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0
    )

    return np.rad2deg(
        np.arccos(dot)
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("U2EYES V5 — FULL FP32 DEPLOYMENT SANITY CHECK")
print("=" * 70)


# ------------------------------------------------------------
# Build
# ------------------------------------------------------------

print(
    "\n[1/4] Loading original and deployment models..."
)

original_model, deploy_model = build_models()

print(
    "Both models loaded."
)


# ------------------------------------------------------------
# Test data
# ------------------------------------------------------------

print(
    "\n[2/4] Loading test set..."
)

df = pd.read_csv(
    TEST_CSV
)

print(
    "Test samples:",
    len(df)
)

if len(df) != 195:

    print(
        "WARNING: expected 195 samples."
    )


# ------------------------------------------------------------
# Arrays
# ------------------------------------------------------------

orig_yaw = []
orig_pitch = []

dep_yaw = []
dep_pitch = []

true_yaw = []
true_pitch = []

orig_h = []
dep_h = []
true_h = []

orig_v = []
dep_v = []
true_v = []

orig_d = []
dep_d = []
true_d = []


# ------------------------------------------------------------
# Output differences
# ------------------------------------------------------------

max_diffs = np.zeros(
    5,
    dtype=np.float64
)

sum_diffs = np.zeros(
    5,
    dtype=np.float64
)

same_h = 0
same_v = 0
same_d = 0


# ============================================================
# FULL TEST SET
# ============================================================

print(
    "\n[3/4] Running both models on all "
    f"{len(df)} test samples..."
)


with torch.no_grad():

    for i, row in df.iterrows():

        left = load_image(
            row["left_image"],
            "left"
        ).unsqueeze(0)

        right = load_image(
            row["right_image"],
            "right"
        ).unsqueeze(0)


        original_output = original_model(
            left,
            right
        )

        deploy_output = deploy_model(
            left,
            right
        )


        # ----------------------------------------------------
        # Convert deploy outputs [B,C,1,1] -> [B,C]
        # ----------------------------------------------------

        deploy_outputs_2d = [

            deploy_output[0][
                :,
                :,
                0,
                0
            ],

            deploy_output[1][
                :,
                :,
                0,
                0
            ],

            deploy_output[2][
                :,
                :,
                0,
                0
            ],

            deploy_output[3][
                :,
                :,
                0,
                0
            ],

            deploy_output[4][
                :,
                :,
                0,
                0
            ]
        ]


        # ----------------------------------------------------
        # Numerical differences
        # ----------------------------------------------------

        for j, (
            a,
            b
        ) in enumerate(
            zip(
                original_output,
                deploy_outputs_2d
            )
        ):

            diff = torch.abs(
                a - b
            ).max().item()

            max_diffs[j] = max(
                max_diffs[j],
                diff
            )

            sum_diffs[j] += float(
                torch.abs(
                    a - b
                ).mean().item()
            )


        # ----------------------------------------------------
        # Original predictions
        # ----------------------------------------------------

        o_reg = original_output[0][
            0
        ].cpu().numpy()

        o_h = int(
            torch.argmax(
                original_output[1],
                dim=1
            )[0].item()
        )

        o_v = int(
            torch.argmax(
                original_output[2],
                dim=1
            )[0].item()
        )

        o_d = int(
            torch.argmax(
                original_output[4],
                dim=1
            )[0].item()
        )


        # ----------------------------------------------------
        # Deployment predictions
        # ----------------------------------------------------

        d_reg = deploy_outputs_2d[0][
            0
        ].cpu().numpy()

        d_h = int(
            torch.argmax(
                deploy_outputs_2d[1],
                dim=1
            )[0].item()
        )

        d_v = int(
            torch.argmax(
                deploy_outputs_2d[2],
                dim=1
            )[0].item()
        )

        d_dir = int(
            torch.argmax(
                deploy_outputs_2d[4],
                dim=1
            )[0].item()
        )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ty = float(
            row["relative_yaw"]
        )

        tp = float(
            row["relative_pitch"]
        )

        th = label_index(
            row["target_horizontal"],
            H_MAP
        )

        tv = label_index(
            row["target_vertical"],
            V_MAP
        )

        td = label_index(
            row["target_direction"],
            D_MAP
        )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        orig_yaw.append(
            o_reg[0]
        )

        orig_pitch.append(
            o_reg[1]
        )

        dep_yaw.append(
            d_reg[0]
        )

        dep_pitch.append(
            d_reg[1]
        )

        true_yaw.append(
            ty
        )

        true_pitch.append(
            tp
        )

        orig_h.append(
            o_h
        )

        dep_h.append(
            d_h
        )

        true_h.append(
            th
        )

        orig_v.append(
            o_v
        )

        dep_v.append(
            d_v
        )

        true_v.append(
            tv
        )

        orig_d.append(
            o_d
        )

        dep_d.append(
            d_dir
        )

        true_d.append(
            td
        )


        if (
            i == 0
            or (i + 1) % 25 == 0
            or i + 1 == len(df)
        ):

            print(
                f"Progress: "
                f"{i + 1}/{len(df)}"
            )


# ============================================================
# NUMPY
# ============================================================

orig_yaw = np.asarray(orig_yaw)
orig_pitch = np.asarray(orig_pitch)

dep_yaw = np.asarray(dep_yaw)
dep_pitch = np.asarray(dep_pitch)

true_yaw = np.asarray(true_yaw)
true_pitch = np.asarray(true_pitch)

orig_h = np.asarray(orig_h)
dep_h = np.asarray(dep_h)
true_h = np.asarray(true_h)

orig_v = np.asarray(orig_v)
dep_v = np.asarray(dep_v)
true_v = np.asarray(true_v)

orig_d = np.asarray(orig_d)
dep_d = np.asarray(dep_d)
true_d = np.asarray(true_d)


# ============================================================
# METRICS
# ============================================================

print(
    "\n[4/4] Computing results..."
)


# ------------------------------------------------------------
# Original FP32
# ------------------------------------------------------------

orig_yaw_mae = np.mean(
    np.abs(
        orig_yaw - true_yaw
    )
)

orig_pitch_mae = np.mean(
    np.abs(
        orig_pitch - true_pitch
    )
)

orig_overall_mae = (
    orig_yaw_mae
    + orig_pitch_mae
) / 2.0

orig_angles = angle_error(
    true_yaw,
    true_pitch,
    orig_yaw,
    orig_pitch
)


# ------------------------------------------------------------
# Deployment FP32
# ------------------------------------------------------------

dep_yaw_mae = np.mean(
    np.abs(
        dep_yaw - true_yaw
    )
)

dep_pitch_mae = np.mean(
    np.abs(
        dep_pitch - true_pitch
    )
)

dep_overall_mae = (
    dep_yaw_mae
    + dep_pitch_mae
) / 2.0

dep_angles = angle_error(
    true_yaw,
    true_pitch,
    dep_yaw,
    dep_pitch
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ORIGINAL V5 FP32"
)

print(
    "=" * 70
)

print(
    f"Yaw MAE       : {orig_yaw_mae:.4f}°"
)

print(
    f"Pitch MAE     : {orig_pitch_mae:.4f}°"
)

print(
    f"Overall MAE   : {orig_overall_mae:.4f}°"
)

print(
    f"Mean Angular  : "
    f"{np.mean(orig_angles):.4f}°"
)

print(
    f"<5°           : "
    f"{np.mean(orig_angles < 5) * 100:.2f}%"
)

print(
    f"<10°          : "
    f"{np.mean(orig_angles < 10) * 100:.2f}%"
)

print(
    f"Horizontal    : "
    f"{classification_accuracy(true_h, orig_h):.2f}%"
)

print(
    f"Vertical      : "
    f"{classification_accuracy(true_v, orig_v):.2f}%"
)

print(
    f"9-Class       : "
    f"{classification_accuracy(true_d, orig_d):.2f}%"
)


print(
    "\n"
    + "=" * 70
)

print(
    "DEPLOYMENT-SAFE FP32"
)

print(
    "=" * 70
)

print(
    f"Yaw MAE       : {dep_yaw_mae:.4f}°"
)

print(
    f"Pitch MAE     : {dep_pitch_mae:.4f}°"
)

print(
    f"Overall MAE   : {dep_overall_mae:.4f}°"
)

print(
    f"Mean Angular  : "
    f"{np.mean(dep_angles):.4f}°"
)

print(
    f"<5°           : "
    f"{np.mean(dep_angles < 5) * 100:.2f}%"
)

print(
    f"<10°          : "
    f"{np.mean(dep_angles < 10) * 100:.2f}%"
)

print(
    f"Horizontal    : "
    f"{classification_accuracy(true_h, dep_h):.2f}%"
)

print(
    f"Vertical      : "
    f"{classification_accuracy(true_v, dep_v):.2f}%"
)

print(
    f"9-Class       : "
    f"{classification_accuracy(true_d, dep_d):.2f}%"
)


# ============================================================
# EQUIVALENCE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "ORIGINAL vs DEPLOYMENT GRAPH"
)

print(
    "=" * 70
)


head_names = [
    "Regression",
    "Horizontal",
    "Vertical",
    "Geometry",
    "Direction"
]


for name, maximum, total in zip(
    head_names,
    max_diffs,
    sum_diffs
):

    mean_diff = total / len(df)

    print(
        f"{name:12s} "
        f"max={maximum:.10f} "
        f"mean={mean_diff:.10f}"
    )


# ------------------------------------------------------------
# Prediction agreement
# ------------------------------------------------------------

h_agreement = classification_accuracy(
    orig_h,
    dep_h
)

v_agreement = classification_accuracy(
    orig_v,
    dep_v
)

d_agreement = classification_accuracy(
    orig_d,
    dep_d
)


print(
    "\nPrediction agreement:"
)

print(
    f"Horizontal : {h_agreement:.2f}%"
)

print(
    f"Vertical   : {v_agreement:.2f}%"
)

print(
    f"9-Class    : {d_agreement:.2f}%"
)


# ============================================================
# FINAL DECISION
# ============================================================

MAX_ALLOWED_DIFF = 1e-3

print(
    "\n"
    + "=" * 70
)


if (
    max(max_diffs) <= MAX_ALLOWED_DIFF
    and h_agreement == 100.0
    and v_agreement == 100.0
    and d_agreement == 100.0
):

    print(
        "PASS — DEPLOYMENT GRAPH PRESERVES V5 BEHAVIOR"
    )

    print(
        f"Maximum numerical difference: "
        f"{max(max_diffs):.10f}"
    )

    print(
        "All discrete gaze predictions agree."
    )

else:

    print(
        "WARNING — DEPLOYMENT GRAPH "
        "REQUIRES INVESTIGATION"
    )

    print(
        f"Maximum numerical difference: "
        f"{max(max_diffs):.10f}"
    )

print(
    "=" * 70
)
