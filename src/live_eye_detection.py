import cv2
import mediapipe as mp

WINDOWS_IP = "192.168.93.1"
STREAM_URL = f"http://{WINDOWS_IP}:5000/video"

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(STREAM_URL)

print("Connecting to Windows camera...")

if not cap.isOpened():
    raise RuntimeError("Could not open Windows camera stream")

print("Camera connected!")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame failed")
        break

    # Convert BGR → RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        h, w = frame.shape[:2]

        # Eye landmark indices
        left_eye_indices = [
            33, 133, 159, 145, 160, 144
        ]

        right_eye_indices = [
            362, 263, 386, 374, 387, 373
        ]

        # Draw eye landmarks
        for idx in left_eye_indices:
            x = int(face.landmark[idx].x * w)
            y = int(face.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        for idx in right_eye_indices:
            x = int(face.landmark[idx].x * w)
            y = int(face.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Calculate eye bounding boxes
        def eye_box(indices):

            xs = [
                int(face.landmark[i].x * w)
                for i in indices
            ]

            ys = [
                int(face.landmark[i].y * h)
                for i in indices
            ]

            xmin = max(0, min(xs) - 20)
            xmax = min(w, max(xs) + 20)

            ymin = max(0, min(ys) - 15)
            ymax = min(h, max(ys) + 15)

            return xmin, ymin, xmax, ymax

        lx1, ly1, lx2, ly2 = eye_box(left_eye_indices)
        rx1, ry1, rx2, ry2 = eye_box(right_eye_indices)

        # Draw bounding boxes
        cv2.rectangle(
            frame,
            (lx1, ly1),
            (lx2, ly2),
            (0, 255, 0),
            2,
        )

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "FACE DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    else:

        cv2.putText(
            frame,
            "NO FACE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Live Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
