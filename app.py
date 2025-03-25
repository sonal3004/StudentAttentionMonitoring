from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

# Load MediaPipe face detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV video capture
cap = cv2.VideoCapture(0)

def detect_attention():
    """Detects if the student is paying attention based on face position."""
    global cap
    if not cap.isOpened():
        return None

    success, frame = cap.read()
    if not success:
        return None

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    attention_score = 5  # Default mid-value
    alert = "Neutral"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extracting eye and face position
            left_eye_x = face_landmarks.landmark[33].x
            right_eye_x = face_landmarks.landmark[263].x
            nose_x = face_landmarks.landmark[1].x

            # Check face orientation
            if nose_x < 0.4:  # Looking left
                attention_score = 2
                alert = "Not Attentive"
            elif nose_x > 0.6:  # Looking right
                attention_score = 2
                alert = "Not Attentive"
            else:
                attention_score = 10  # Looking straight
                alert = "Attentive"

    return {'attention_score': attention_score, 'alert': alert}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    def generate():
        global cap
        while True:
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_attention')
def get_attention():
    """Returns the real-time attention score as JSON."""
    try:
        data = detect_attention()
        if data:
            return jsonify(data)
        else:
            return jsonify({'error': 'Failed to capture frame'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """Serves the favicon.ico file."""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    app.run(debug=True)
