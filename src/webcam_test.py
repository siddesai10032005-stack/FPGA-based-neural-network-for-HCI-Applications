import cv2

cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Webcam opened successfully.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame")
        break

    cv2.imshow(
        "U2Eyes Webcam Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
