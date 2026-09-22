import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# CONFIG
# ============================================================

STREAM_URL = "http://192.168.93.1:5000/video"


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


# MediaPipe iris landmark groups
#
# 468-472 = one iris
# 473-477 = other iris

IRIS_A = [
    468, 469, 470, 471, 472
]

IRIS_B = [
    473, 474, 475, 476, 477
]


# ============================================================
# FIND IRIS CENTRE
# ============================================================

def get_landmark_center(face, indices):

    xs = [
        face.landmark[i].x
        for i in indices
    ]

    ys = [
        face.landmark[i].y
        for i in indices
    ]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys)
    )


# ============================================================
# GET EYE BOUNDS
# ============================================================

def get_eye_bounds(face, indices):

    xs = [
        face.landmark[i].x
        for i in indices
    ]

    ys = [
        face.landmark[i].y
        for i in indices
    ]

    return (
        min(xs),
        max(xs),
        min(ys),
        max(ys)
    )


# ============================================================
# NORMALIZE IRIS POSITION
# ============================================================

def normalize_iris(
    iris_x,
    iris_y,
    eye_indices,
    face
):

    min_x, max_x, min_y, max_y = \
        get_eye_bounds(
            face,
            eye_indices
        )

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

    return x, y


# ============================================================
# DETERMINE WHICH IRIS BELONGS TO WHICH EYE
# ============================================================

def get_both_iris_centers(face):

    a_x, a_y = get_landmark_center(
        face,
        IRIS_A
    )

    b_x, b_y = get_landmark_center(
        face,
        IRIS_B
    )

    # Face coordinate system:
    # smaller x = image left
    # larger x  = image right
    #
    # After unmirroring:
    # anatomical right eye is image-left
    # anatomical left eye is image-right

    if a_x < b_x:

        right_iris = (a_x, a_y)
        left_iris = (b_x, b_y)

    else:

        right_iris = (b_x, b_y)
        left_iris = (a_x, a_y)

    return left_iris, right_iris


# ============================================================
# CLASSIFY
# ============================================================

def classify_iris(x, y):

    # Horizontal

    if x < 0.38:
        horizontal = "LEFT"

    elif x > 0.62:
        horizontal = "RIGHT"

    else:
        horizontal = "CENTER"

    # Vertical

    if y < 0.38:
        vertical = "UP"

    elif y > 0.62:
        vertical = "DOWN"

    else:
        vertical = "CENTER"

    # Combine

    if (
        horizontal == "CENTER"
        and vertical == "CENTER"
    ):

        return "CENTER"

    if vertical == "CENTER":
        return horizontal

    if horizontal == "CENTER":
        return vertical

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
        "Could not open Windows camera stream"
    )

print("Camera connected!")
print()
print("IMPORTANT:")
print("Camera is being unmirrored.")
print()
print("Look:")
print("LEFT")
print("RIGHT")
print("UP")
print("DOWN")
print("CENTER")
print()
print("Press Q to quit.")


# ============================================================
# LIVE LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # ========================================================
    # UNMIRROR CAMERA
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )

    height, width = frame.shape[:2]

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(
        rgb
    )

    # ========================================================
    # NO FACE
    # ========================================================

    if not results.multi_face_landmarks:

        cv2.putText(
            frame,
            "FACE: NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "MEDIA PIPE IRIS TEST",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # ========================================================
    # FACE
    # ========================================================

    face = results.multi_face_landmarks[0]

    # ========================================================
    # IRIS CENTERS
    # ========================================================

    left_iris, right_iris = \
        get_both_iris_centers(face)

    left_x_raw, left_y_raw = left_iris
    right_x_raw, right_y_raw = right_iris

    # ========================================================
    # NORMALIZED POSITIONS
    # ========================================================

    left_x, left_y = normalize_iris(
        left_x_raw,
        left_y_raw,
        LEFT_EYE,
        face
    )

    right_x, right_y = normalize_iris(
        right_x_raw,
        right_y_raw,
        RIGHT_EYE,
        face
    )

    # Clamp for display
    left_x = np.clip(
        left_x,
        0.0,
        1.0
    )

    left_y = np.clip(
        left_y,
        0.0,
        1.0
    )

    right_x = np.clip(
        right_x,
        0.0,
        1.0
    )

    right_y = np.clip(
        right_y,
        0.0,
        1.0
    )

    # ========================================================
    # AVERAGE
    # ========================================================

    avg_x = (
        left_x + right_x
    ) / 2.0

    avg_y = (
        left_y + right_y
    ) / 2.0

    direction = classify_iris(
        avg_x,
        avg_y
    )

    # ========================================================
    # DRAW IRIS CENTERS
    # ========================================================

    for x, y in [
        left_iris,
        right_iris
    ]:

        px = int(
            x * width
        )

        py = int(
            y * height
        )

        cv2.circle(
            frame,
            (px, py),
            5,
            (0, 0, 255),
            -1
        )

    # ========================================================
    # DRAW EYE LANDMARKS
    # ========================================================

    for index in (
        LEFT_EYE + RIGHT_EYE
    ):

        landmark = face.landmark[index]

        px = int(
            landmark.x * width
        )

        py = int(
            landmark.y * height
        )

        cv2.circle(
            frame,
            (px, py),
            2,
            (0, 255, 0),
            -1
        )

    # ========================================================
    # TEXT
    # ========================================================

    cv2.putText(
        frame,
        "MEDIA PIPE IRIS TEST",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"IRIS DIRECTION: {direction}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"AVG X: {avg_x:.3f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"AVG Y: {avg_y:.3f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"LEFT  X={left_x:.3f} Y={left_y:.3f}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"RIGHT X={right_x:.3f} Y={right_y:.3f}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "RED = IRIS CENTER",
        (20, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "MEDIA PIPE IRIS TEST",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

face_mesh.close()

cv2.destroyAllWindows()

print()
print("MediaPipe iris test stopped.")
