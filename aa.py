import cv2
import dlib
import numpy as np
import time
import threading
import queue
import os

# Constants
FACE_MODEL = "mmod_human_face_detector.dat"
PREDICTOR_MODEL = "shape_predictor_68_face_landmarks.dat"
MOUTH_POINTS = list(range(48, 68))
MAX_FPS = 30

# Check model files
if not os.path.exists(FACE_MODEL) or not os.path.exists(PREDICTOR_MODEL):
    print("[ERROR] Model files missing.")
    exit()

# Load models
face_detector = dlib.cnn_face_detection_model_v1(FACE_MODEL)
predictor = dlib.shape_predictor(PREDICTOR_MODEL)

# Initialize variables
text = ""
last_letter_time = 0
frame_queue = queue.Queue(maxsize=1)
running = True

def get_lip_features(landmarks):
    mouth = np.array([landmarks[i] for i in MOUTH_POINTS])
    top_lip = np.mean(mouth[2:4], axis=0)
    bottom_lip = np.mean(mouth[8:10], axis=0)
    left_corner = mouth[0]
    right_corner = mouth[6]
    openness = np.linalg.norm(top_lip - bottom_lip)
    width = np.linalg.norm(left_corner - right_corner)
    roundness = openness / (width + 1e-5)
    return roundness, openness, width

def classify_vowel(roundness, openness):
    if openness < 5:
        return "M"
    elif roundness > 0.4:
        return "O"
    elif roundness < 0.25 and openness > 7:
        return "A"
    else:
        return "-"

def capture_frames(cap):
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        if not frame_queue.full():
            frame_queue.put(frame)

# Setup camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, MAX_FPS)

# Start frame capture thread
thread = threading.Thread(target=capture_frames, args=(cap,))
thread.daemon = True
thread.start()

# Main loop
while True:
    if frame_queue.empty():
        continue

    frame = frame_queue.get()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        rect = face.rect
        shape = predictor(gray, rect)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        roundness, openness, width = get_lip_features(landmarks)
        letter = classify_vowel(roundness, openness)

        if letter != "-" and time.time() - last_letter_time > 1:
            text += letter
            last_letter_time = time.time()

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Draw output text
    cv2.rectangle(frame, (10, 430), (630, 470), (255, 255, 255), -1)
    cv2.putText(frame, text, (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Silent Speech Keyboard (Optimized)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running = False
        break
    elif key == ord('c'):
        text = ""

cap.release()
cv2.destroyAllWindows()
