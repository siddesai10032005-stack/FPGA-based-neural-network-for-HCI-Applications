import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import os
import time


# ============================================================
# CONFIG
# ============================================================

STREAM_URL = "http://192.168.93.1:5000/video"
MODEL_PATH = "models/best_multitask_v5.pth"
CALIBRATION_FILE = "models/v5_webcam_calibration.npz"

DEVICE = torch.device("cpu")

IMG_W = 128
IMG_H = 64

# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

# EMA smoothing:
# higher alpha = more responsive
# lower alpha  = smoother but slower
ALPHA = 0.30

smooth_yaw = None
smooth_pitch = None


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

        fusion_features = self.fusion(fusion_input)

        direction = self.direction_head(
            fusion_features
        )

        return {
            "regression": regression,
            "horizontal": horizontal,
            "vertical": vertical,
            "direction": direction,
            "geometry": geometry
        }


# ============================================================
# LOAD V5
# ============================================================

print("Loading V5...")

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

print("V5 loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# EYE LANDMARKS
# ============================================================

RIGHT_EYE = [
    33, 133, 159, 145, 160, 144
]

LEFT_EYE = [
    362, 263, 386, 374, 387, 373
]

IRIS_A = [
    468, 469, 470, 471, 472
]

IRIS_B = [
    473, 474, 475, 476, 477
]


# ============================================================
# LABELS
# ============================================================

DIRECTION_CLASSES = [
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

HORIZONTAL_CLASSES = [
    "LEFT",
    "CENTER",
    "RIGHT"
]

VERTICAL_CLASSES = [
    "DOWN",
    "CENTER",
    "UP"
]


# ============================================================
# EYE BOX
# ============================================================

def get_eye_box(
    face,
    indices,
    width,
    height
):

    xs = [
        int(face.landmark[i].x * width)
        for i in indices
    ]

    ys = [
        int(face.landmark[i].y * height)
        for i in indices
    ]

    margin_x = 28
    margin_y = 20

    x1 = max(
        0,
        min(xs) - margin_x
    )

    x2 = min(
        width,
        max(xs) + margin_x
    )

    y1 = max(
        0,
        min(ys) - margin_y
    )

    y2 = min(
        height,
        max(ys) + margin_y
    )

    return x1, y1, x2, y2


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_eye(
    frame,
    box
):

    x1, y1, x2, y2 = box

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    crop = cv2.resize(
        crop,
        (IMG_W, IMG_H),
        interpolation=cv2.INTER_AREA
    )

    crop = crop.astype(
        np.float32
    ) / 255.0

    crop = np.transpose(
        crop,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        crop
    ).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# IRIS CENTER
# ============================================================

def iris_center(
    face,
    indices
):

    xs = [
        face.landmark[i].x
        for i in indices
    ]

    ys = [
        face.landmark[i].y
        for i in indices
    ]

    return (
        float(np.mean(xs)),
        float(np.mean(ys))
    )


# ============================================================
# IRIS NORMALIZATION
# ============================================================

def normalize_iris(
    face,
    iris_x,
    iris_y,
    eye_indices
):

    xs = [
        face.landmark[i].x
        for i in eye_indices
    ]

    ys = [
        face.landmark[i].y
        for i in eye_indices
    ]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    width = max_x - min_x
    height = max_y - min_y

    if width < 1e-6:
        width = 1e-6

    if height < 1e-6:
        height = 1e-6

    x = (
        iris_x - min_x
    ) / width

    y = (
        iris_y - min_y
    ) / height

    return (
        float(np.clip(x, 0, 1)),
        float(np.clip(y, 0, 1))
    )


# ============================================================
# GET REAL CAMERA FEATURES
# ============================================================

def get_camera_features(
    face
):

    a = iris_center(
        face,
        IRIS_A
    )

    b = iris_center(
        face,
        IRIS_B
    )

    # Anatomical right eye is image-left
    if a[0] < b[0]:

        right_iris = a
        left_iris = b

    else:

        right_iris = b
        left_iris = a

    left_x, left_y = normalize_iris(
        face,
        left_iris[0],
        left_iris[1],
        LEFT_EYE
    )

    right_x, right_y = normalize_iris(
        face,
        right_iris[0],
        right_iris[1],
        RIGHT_EYE
    )

    avg_x = (
        left_x + right_x
    ) / 2.0

    avg_y = (
        left_y + right_y
    ) / 2.0

    return (
        left_x,
        left_y,
        right_x,
        right_y,
        avg_x,
        avg_y
    )


# ============================================================
# V5 + MEDIAPIPE
# ============================================================

def run_system(frame):

    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(
        rgb
    )

    if not results.multi_face_landmarks:
        return None

    face = results.multi_face_landmarks[0]

    left_box = get_eye_box(
        face,
        LEFT_EYE,
        width,
        height
    )

    right_box = get_eye_box(
        face,
        RIGHT_EYE,
        width,
        height
    )

    left_tensor = preprocess_eye(
        frame,
        left_box
    )

    right_tensor = preprocess_eye(
        frame,
        right_box
    )

    if (
        left_tensor is None
        or right_tensor is None
    ):
        return None

    # --------------------------------------------------------
    # V5
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            left_tensor,
            right_tensor
        )

    gaze = (
        output["regression"][0]
        .cpu()
        .numpy()
    )

    raw_yaw = float(gaze[0])
    raw_pitch = float(gaze[1])

    h_probs = torch.softmax(
        output["horizontal"],
        dim=1
    )[0].cpu().numpy()

    v_probs = torch.softmax(
        output["vertical"],
        dim=1
    )[0].cpu().numpy()

    d_probs = torch.softmax(
        output["direction"],
        dim=1
    )[0].cpu().numpy()

    h_idx = int(np.argmax(h_probs))
    v_idx = int(np.argmax(v_probs))
    d_idx = int(np.argmax(d_probs))

    # --------------------------------------------------------
    # MediaPipe iris
    # --------------------------------------------------------

    (
        left_x,
        left_y,
        right_x,
        right_y,
        iris_x,
        iris_y
    ) = get_camera_features(face)

    return {
        "raw_yaw": raw_yaw,
        "raw_pitch": raw_pitch,

        "h_label": HORIZONTAL_CLASSES[h_idx],
        "h_conf": float(h_probs[h_idx]),

        "v_label": VERTICAL_CLASSES[v_idx],
        "v_conf": float(v_probs[v_idx]),

        "direct_direction": DIRECTION_CLASSES[d_idx],
        "direct_conf": float(d_probs[d_idx]),

        "iris_x": iris_x,
        "iris_y": iris_y,

        "left_x": left_x,
        "left_y": left_y,

        "right_x": right_x,
        "right_y": right_y,

        "left_box": left_box,
        "right_box": right_box
    }


# ============================================================
# CALIBRATION TARGETS
# ============================================================

TARGETS = [

    ("UP_LEFT",       -18.0,  10.0),
    ("UP_CENTER",       0.0,  10.0),
    ("UP_RIGHT",       18.0,  10.0),

    ("CENTER_LEFT",   -18.0,   0.0),
    ("CENTER_CENTER",  0.0,   0.0),
    ("CENTER_RIGHT",   18.0,   0.0),

    ("DOWN_LEFT",     -18.0, -10.0),
    ("DOWN_CENTER",    0.0,  -10.0),
    ("DOWN_RIGHT",    18.0,  -10.0)
]


# ============================================================
# CALIBRATION VARIABLES
# ============================================================

calibration_samples = []

calibration_index = 0

calibration_mode = False

calibration_start = None

CALIBRATION_TIME = 3.0

calibration_model = None


# ============================================================
# LOAD EXISTING CALIBRATION
# ============================================================

if os.path.exists(
    CALIBRATION_FILE
):

    try:

        data = np.load(
            CALIBRATION_FILE
        )

        calibration_model = data["coef"]

        print(
            "Saved webcam calibration loaded."
        )

    except Exception as e:

        print(
            "Could not load calibration:",
            e
        )


# ============================================================
# FIT HYBRID CALIBRATION
# ============================================================

def fit_calibration(
    samples
):

    X = []
    Y = []

    for sample in samples:

        (
            raw_yaw,
            raw_pitch,
            iris_x,
            iris_y,
            target_yaw,
            target_pitch
        ) = sample

        # Features:
        #
        # V5 yaw
        # V5 pitch
        # MediaPipe iris X
        # MediaPipe iris Y
        # bias

        X.append([
            raw_yaw,
            raw_pitch,
            iris_x,
            iris_y,
            1.0
        ])

        Y.append([
            target_yaw,
            target_pitch
        ])

    X = np.asarray(
        X,
        dtype=np.float32
    )

    Y = np.asarray(
        Y,
        dtype=np.float32
    )

    # Ridge regression for stability
    regularization = 0.01

    A = (
        X.T @ X
        + regularization * np.eye(
            X.shape[1]
        )
    )

    B = X.T @ Y

    coef = np.linalg.solve(
        A,
        B
    )

    return coef


# ============================================================
# APPLY CALIBRATION
# ============================================================

def apply_calibration(
    calibration,
    raw_yaw,
    raw_pitch,
    iris_x,
    iris_y
):

    X = np.array([
        raw_yaw,
        raw_pitch,
        iris_x,
        iris_y,
        1.0
    ])

    result = X @ calibration

    return (
        float(result[0]),
        float(result[1])
    )


# ============================================================
# CLASSIFY FINAL GAZE
# ============================================================

def classify_gaze(
    yaw,
    pitch
):

    if yaw < -7:
        horizontal = "LEFT"

    elif yaw > 7:
        horizontal = "RIGHT"

    else:
        horizontal = "CENTER"

    if pitch > 4:
        vertical = "UP"

    elif pitch < -4:
        vertical = "DOWN"

    else:
        vertical = "CENTER"

    if (
        horizontal == "CENTER"
        and vertical == "CENTER"
    ):
        return "CENTER_CENTER"

    if vertical == "CENTER":
        return "CENTER_" + horizontal

    if horizontal == "CENTER":
        return vertical + "_CENTER"

    return vertical + "_" + horizontal


# ============================================================
# CAMERA
# ============================================================

print()
print("Connecting to Windows camera...")

cap = cv2.VideoCapture(
    STREAM_URL
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open camera stream."
    )

print("Camera connected.")
print()
print("================================")
print("V5 REAL CAMERA HYBRID SYSTEM")
print("================================")
print()
print("Temporal smoothing: ENABLED")
print(f"EMA alpha: {ALPHA}")
print()
print("C = start calibration")
print("R = remove saved calibration")
print("Q = quit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # ========================================================
    # UNMIRROR
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )

    result = run_system(
        frame
    )

    # ========================================================
    # NO FACE
    # ========================================================

    if result is None:

        # Reset smoothing when face disappears
        smooth_yaw = None
        smooth_pitch = None

        cv2.putText(
            frame,
            "PERSON: NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:

        # ----------------------------------------------------
        # DRAW EYE BOXES
        # ----------------------------------------------------

        lx1, ly1, lx2, ly2 = \
            result["left_box"]

        rx1, ry1, rx2, ry2 = \
            result["right_box"]

        cv2.rectangle(
            frame,
            (lx1, ly1),
            (lx2, ly2),
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # RAW V5
        # ----------------------------------------------------

        raw_yaw = result["raw_yaw"]
        raw_pitch = result["raw_pitch"]

        # ----------------------------------------------------
        # CALIBRATED GAZE
        # ----------------------------------------------------

        if calibration_model is not None:

            calibrated_yaw, calibrated_pitch = \
                apply_calibration(
                    calibration_model,
                    raw_yaw,
                    raw_pitch,
                    result["iris_x"],
                    result["iris_y"]
                )

        else:

            calibrated_yaw = raw_yaw
            calibrated_pitch = raw_pitch

        # ----------------------------------------------------
        # TEMPORAL SMOOTHING
        # ----------------------------------------------------

        if smooth_yaw is None:

            smooth_yaw = calibrated_yaw
            smooth_pitch = calibrated_pitch

        else:

            smooth_yaw = (
                ALPHA * calibrated_yaw
                + (1.0 - ALPHA) * smooth_yaw
            )

            smooth_pitch = (
                ALPHA * calibrated_pitch
                + (1.0 - ALPHA) * smooth_pitch
            )

        # ----------------------------------------------------
        # FINAL GAZE
        # ----------------------------------------------------

        final_yaw = smooth_yaw
        final_pitch = smooth_pitch

        final_direction = classify_gaze(
            final_yaw,
            final_pitch
        )

        # ----------------------------------------------------
        # MAIN DISPLAY
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "V5 + MEDIAPIPE + CALIBRATION + SMOOTHING",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"GAZE: {final_direction}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Final Yaw: {final_yaw:+.1f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Final Pitch: {final_pitch:+.1f}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # CALIBRATED RAW VALUE
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Calibrated: "
            f"{calibrated_yaw:+.1f}, "
            f"{calibrated_pitch:+.1f}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 255, 255),
            2
        )

        # ----------------------------------------------------
        # V5 INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"V5 raw: {result['direct_direction']} "
            f"{result['direct_conf'] * 100:.0f}%",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"V5 H: {result['h_label']} "
            f"{result['h_conf'] * 100:.0f}%",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"V5 V: {result['v_label']} "
            f"{result['v_conf'] * 100:.0f}%",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # IRIS INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Iris X: {result['iris_x']:.3f}",
            (20, 305),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Iris Y: {result['iris_y']:.3f}",
            (20, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 255),
            2
        )

        # ----------------------------------------------------
        # CALIBRATION MODE
        # ----------------------------------------------------

        if calibration_mode:

            name, target_yaw, target_pitch = \
                TARGETS[
                    calibration_index
                ]

            elapsed = (
                time.time()
                - calibration_start
            )

            remaining = max(
                0,
                CALIBRATION_TIME - elapsed
            )

            cv2.putText(
                frame,
                "CALIBRATING",
                (20, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"LOOK: {name}",
                (20, 420),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                3
            )

            cv2.putText(
                frame,
                f"POINT {calibration_index + 1}/9",
                (20, 455),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Hold: {remaining:.1f}s",
                (20, 490),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Wait first 1 second for movement to settle
            # ------------------------------------------------

            if elapsed > 1.0:

                calibration_samples.append(
                    (
                        raw_yaw,
                        raw_pitch,
                        result["iris_x"],
                        result["iris_y"],
                        target_yaw,
                        target_pitch
                    )
                )

            # ------------------------------------------------
            # Next target
            # ------------------------------------------------

            if elapsed >= CALIBRATION_TIME:

                print(
                    f"Captured {name}: "
                    f"{len(calibration_samples)} samples"
                )

                calibration_index += 1

                if calibration_index >= len(
                    TARGETS
                ):

                    # ----------------------------------------
                    # FIT
                    # ----------------------------------------

                    calibration_model = \
                        fit_calibration(
                            calibration_samples
                        )

                    np.savez(
                        CALIBRATION_FILE,
                        coef=calibration_model
                    )

                    calibration_mode = False

                    # Reset smoothing after new calibration
                    smooth_yaw = None
                    smooth_pitch = None

                    print()
                    print(
                        "================================"
                    )
                    print(
                        "CALIBRATION COMPLETE"
                    )
                    print(
                        "================================"
                    )
                    print(
                        "Saved:",
                        CALIBRATION_FILE
                    )
                    print()

                else:

                    calibration_start = time.time()

        else:

            if calibration_model is not None:

                cv2.putText(
                    frame,
                    "CALIBRATION: ON",
                    (20, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "SMOOTHING: ON",
                    (20, 410),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Press C = calibrate",
                    (20, 380),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2
                )

                cv2.putText(
                    frame,
                    "SMOOTHING: ON",
                    (20, 410),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "V5 REAL CAMERA",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # START CALIBRATION
    # ========================================================

    if key == ord("c"):

        calibration_samples = []

        calibration_index = 0

        calibration_start = time.time()

        calibration_mode = True

        calibration_model = None

        # Reset smoothing
        smooth_yaw = None
        smooth_pitch = None

        print()
        print(
            "================================"
        )
        print(
            "NEW CALIBRATION STARTED"
        )
        print(
            "================================"
        )
        print()

    # ========================================================
    # REMOVE CALIBRATION
    # ========================================================

    if key == ord("r"):

        calibration_model = None

        # Reset smoothing
        smooth_yaw = None
        smooth_pitch = None

        if os.path.exists(
            CALIBRATION_FILE
        ):

            os.remove(
                CALIBRATION_FILE
            )

        print(
            "Calibration deleted."
        )

    # ========================================================
    # QUIT
    # ========================================================

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

face_mesh.close()

cv2.destroyAllWindows()

print()
print("Stopped.")
