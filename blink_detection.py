import cv2
import dlib
import numpy as np
import pyttsx3
from scipy.spatial import distance as dist

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Load pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib's official repo

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    EAR = (A + B) / (2.0 * C)
    return EAR

# Eye landmark indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Blink threshold settings
EAR_THRESHOLD = 0.25  # Below this value, the eye is considered closed
BLINK_FRAME_THRESHOLD = 3  # Number of consecutive frames to consider a blink
NO_BLINK_ALERT_TIME = 100  # Time (frames) before generating alert

cap = cv2.VideoCapture(0)
frame_count = 0
blink_count = 0
alert_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get eye coordinates
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])

        # Compute EAR for both eyes
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw eye contours
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        if avg_EAR < EAR_THRESHOLD:
            blink_count += 1
        else:
            if blink_count >= BLINK_FRAME_THRESHOLD:
                blink_count = 0  # Reset blink counter
            frame_count = 0  # Reset alert counter
            alert_triggered = False

    # Alert if no blink detected for too long
    frame_count += 1
    if frame_count > NO_BLINK_ALERT_TIME and not alert_triggered:
        engine.say("Please blink your eyes")
        engine.runAndWait()
        alert_triggered = True  # Avoid multiple alerts

    # Display the frame
    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
