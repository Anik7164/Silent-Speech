import cv2
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import time

# Setup
MODEL_PATH = "mouth_knn_model.pkl"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
MOUTH_POINTS = list(range(48, 68))

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Classifier and buffers
if os.path.exists(MODEL_PATH):
    knn = joblib.load(MODEL_PATH)
    print("[INFO] Loaded trained model.")
else:
    knn = KNeighborsClassifier(n_neighbors=3)
    print("[INFO] No model found, starting new session.")

trained_data, trained_labels = [], []

# Feature extraction
def get_mouth_descriptor(landmarks):
    mouth = np.array([landmarks[i] for i in MOUTH_POINTS])
    center = np.mean(mouth, axis=0)
    normalized = (mouth - center).flatten()
    return normalized / (np.linalg.norm(normalized) + 1e-8)

# Train classifier
def train_model():
    if len(trained_data) >= 1:
        knn.fit(trained_data, trained_labels)
        joblib.dump(knn, MODEL_PATH)
        print("[TRAINED] Model saved.")
        return True
    else:
        print("[WARN] Not enough samples to train.")
        return False

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

text = ""
last_letter_time = 0
training_mode = False
current_letter = ""
feedback_text = ""
feedback_timer = 0

print("[INFO] Press A-Z to record samples | T = train | C = clear | Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    descriptor = None

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        descriptor = get_mouth_descriptor(landmarks)

        # Predict letter
        if hasattr(knn, "classes_") and len(knn.classes_) > 0 and not training_mode:
            prediction = knn.predict([descriptor])[0]
            confidence = knn.predict_proba([descriptor]).max()
            if confidence > 0.85 and time.time() - last_letter_time > 1:
                text += prediction
                last_letter_time = time.time()
                print(f"[PREDICT] {prediction} ({confidence:.2f})")
                feedback_text = f"Predicted: {prediction}"
                feedback_timer = time.time()

        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Highlight mouth area
        mouth_pts = np.array([landmarks[i] for i in MOUTH_POINTS], np.int32)
        cv2.polylines(frame, [mouth_pts], True, (0, 255, 255), 1)

    # Draw output box
    cv2.rectangle(frame, (10, 430), (630, 470), (255, 255, 255), -1)
    cv2.putText(frame, text, (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Show training mode status
    if training_mode:
        cv2.putText(frame, f"TRAINING MODE: {current_letter}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show feedback message (lasts 2 seconds)
    if time.time() - feedback_timer < 2 and feedback_text:
        cv2.putText(frame, feedback_text, 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Show instructions
    cv2.putText(frame, "A-Z: Record | T: Train | C: Clear | Q: Quit", 
               (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Silent Speech Keyboard", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        text = ""
    elif key == ord('t'):
        if train_model():
            feedback_text = "Model trained successfully!"
            feedback_timer = time.time()
        else:
            feedback_text = "Not enough samples to train!"
            feedback_timer = time.time()
    elif 97 <= key <= 122:  # a-z
        current_letter = chr(key).upper()
        training_mode = True
        feedback_text = f"Recording: {current_letter}"
        feedback_timer = time.time()
    elif key == 32 and training_mode and descriptor is not None:  # Space to capture
        trained_data.append(descriptor)
        trained_labels.append(current_letter)
        feedback_text = f"Recorded: {current_letter}"
        feedback_timer = time.time()
        print(f"[RECORDED] {current_letter}")
        training_mode = False
    elif key == 27:  # ESC to cancel recording
        training_mode = False
        feedback_text = "Recording cancelled"
        feedback_timer = time.time()

cap.release()
cv2.destroyAllWindows()