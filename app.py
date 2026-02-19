import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from threading import Thread
from openai import OpenAI 

# Import your new exercise logic file
import exercise_logic

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hackathon_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- GROQ SETUP ---
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
)

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global trackers
active_exercise = "pushup" # Default
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
    global active_exercise, rep_count, current_stage
    env = request.args.get('env', 'gym') 
    
    # Read which exercise was clicked on the dashboard
    active_exercise = request.args.get('ex', 'pushup')
    
    # Reset counters for the new exercise
    rep_count = 0
    current_stage = "up"
    
    return render_template('index.html', env=env, exercise=active_exercise)

# --- AI VISION LOGIC ---
def process_webcam():
    global current_stage, rep_count, active_exercise
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
                
                # --- Send to the logic module ---
                rep_count, current_stage, status, play_audio = exercise_logic.detect_exercise(
                    active_exercise, landmarks, current_stage, rep_count
                )
                
                # --- Trigger events based on response ---
                if play_audio:
                    socketio.emit('play_audio', {'file': 'good'})

                socketio.emit('update_vr', {
                    'reps': rep_count,
                    'status': status
                })
            except Exception as e:
                pass
        
        cv2.imshow('PhysioVR Vision', image) 
        if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- CHAT LOGIC (Unchanged) ---
@socketio.on('send_chat')
def handle_chat(data):
    user_text = data['message']
    print(f"User asked: {user_text}")

    try:
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