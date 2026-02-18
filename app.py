import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from threading import Thread
from openai import OpenAI 

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hackathon_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- GROQ SETUP (Updated Model) ---
# 🔴 PASTE YOUR GROQ KEY BELOW 🔴
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

current_stage = "up"
rep_count = 0

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/coach')
def coach():
    return render_template('coach.html')

@app.route('/vr')
def vr_session():
    env = request.args.get('env', 'gym') 
    exercise = request.args.get('ex', 'pushup')
    return render_template('index.html', env=env, exercise=exercise)

# --- AI VISION LOGIC ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def process_webcam():
    global current_stage, rep_count
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                landmarks = results.pose_landmarks.landmark
                # Right Arm
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                angle = calculate_angle(shoulder, elbow, wrist)
                
                if angle < 90: current_stage = "down"
                if angle > 160 and current_stage == 'down':
                    current_stage = "up"
                    rep_count += 1
                    socketio.emit('play_audio', {'file': 'good'})

                socketio.emit('update_vr', {
                    'reps': rep_count,
                    'status': "Push!" if current_stage == "up" else "Up!"
                })
            except:
                pass
        
        # cv2.imshow('PhysioVR Vision', image) 
        if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- CHAT LOGIC ---
@socketio.on('send_chat')
def handle_chat(data):
    user_text = data['message']
    print(f"User asked: {user_text}")

    try:
        # UPDATED MODEL NAME BELOW
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": "You are PhysioVR Coach. Keep answers under 50 words."},
                {"role": "user", "content": user_text}
            ]
        )
        ai_reply = response.choices[0].message.content
        print(f"AI replied: {ai_reply}")
        
    except Exception as e:
        print(f"Groq Error: {e}")
        ai_reply = "I'm having trouble connecting to the cloud. Please check your internet."

    emit('receive_chat', {'role': 'ai', 'text': ai_reply})

if __name__ == '__main__':
    t = Thread(target=process_webcam)
    t.daemon = True
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)