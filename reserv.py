"""from flask import Flask, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Load OpenCV face and eye classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_faces():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 0:
                cv2.putText(frame, "ALERT: Eyes Closed!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except:
                pass
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def home():
    return "Face & Eye Sentiment Detection Running!"

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)"""


"""   


from flask import Flask, Response
import cv2
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

# Load OpenCV face and eye classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_faces():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Check if eyes are closed using EAR (Eye Aspect Ratio)
            eye_status = "Open"
            if len(eyes) >= 2:  # Ensure both eyes are detected
                ear_values = []
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    # Calculate EAR for each eye
                    eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                    _, thresh = cv2.threshold(eye_region, 10, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(contour)
                        ear_values.append(area / (ew * eh))
                
                # Average EAR value for both eyes
                if ear_values:
                    avg_ear = np.mean(ear_values)
                    if avg_ear < 0.2:  # Threshold for closed eyes
                        eye_status = "Closed"
            
            # Display eye status
            cv2.putText(frame, f"Eyes: {eye_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            try:
                # Analyze emotion only if the face is large enough
                if w > 100 and h > 100:
                    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error analyzing emotion: {e}")

        # Encode and yield the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return "Face & Eye Sentiment Detection Running!"

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)"""

"""from flask import Flask, Response
import cv2
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

# Load OpenCV face and eye classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Store the last detected emotion to avoid random jumps
emotion_history = []
MAX_HISTORY = 5  # Adjust for smoother changes

def smooth_emotion(emotion):
    
    global emotion_history
    emotion_history.append(emotion)
    if len(emotion_history) > MAX_HISTORY:
        emotion_history.pop(0)
    return max(set(emotion_history), key=emotion_history.count)

def detect_faces():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(eyes) < 2:
                cv2.putText(frame, "ALERT: Eyes Closed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            try:
                # Analyze emotion only on the detected face for better accuracy
                analysis = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)
                emotion = smooth_emotion(analysis[0]['dominant_emotion'])
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error analyzing emotion: {e}")

        # Encode and yield the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return "Face & Eye Sentiment Detection Running!"

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
"""
"""
from flask import Flask, Response
import cv2
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

# Load OpenCV face and eye classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Store the last detected emotion to avoid random jumps
emotion_history = []
MAX_HISTORY = 10  # Increase history for smoother changes

def smooth_emotion(emotion):
    global emotion_history
    emotion_history.append(emotion)
    if len(emotion_history) > MAX_HISTORY:
        emotion_history.pop(0)
    return max(set(emotion_history), key=emotion_history.count)

def detect_faces():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(eyes) < 2:
                cv2.putText(frame, "ALERT: Eyes Closed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Limit detection to two eye boxes only
                eyes = sorted(eyes, key=lambda e: e[0])[:2]
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            try:
                # Analyze emotion only on the detected face for better accuracy
                analysis = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)
                emotion = smooth_emotion(analysis[0]['dominant_emotion'])
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error analyzing emotion: {e}")

        # Encode and yield the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return "Face & Eye Sentiment Detection Running!"

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
"""