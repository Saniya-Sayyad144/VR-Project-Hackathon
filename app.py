import os
import cv2
import mediapipe as mp
import numpy as np
import time
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect
from flask_socketio import SocketIO, emit
from threading import Thread
from openai import OpenAI 
from dotenv import load_dotenv

import exercise_logic
import mysql_helper
import bcrypt
import jwt
import jwt_handler
from login_required import login_required

load_dotenv()

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "dev_secret")

# JWT secret (use Flask secret key)
JWT_SECRET = app.config['SECRET_KEY']
JWT_ALGORITHM = 'HS256'

socketio = SocketIO(app, cors_allowed_origins="*")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- NEW STATE MEMORY ---
active_exercise = "pushup"
is_workout_active = False
workout_mode = 'reps'  
target_reps = 10
start_time = 0
target_duration = 0 

# This dictionary replaces the old isolated variables
exercise_state = {
    'stage': 'up',
    'reps': 0,
    'lowest_angle': 180,
    'status': 'Get into position'
}

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('home.html', auth_check=True)

@app.route('/login-page', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/register-page', methods=['GET'])
def register_page():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not username or not email or not password:
        return jsonify({'error': 'Missing fields'}), 400

    # Hash the password
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    hashed_str = hashed.decode('utf-8')

    try:
        conn = mysql_helper.get_mysql_connection()
        cursor = conn.cursor(prepared=True)
        sql = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
        cursor.execute(sql, (username, email, hashed_str))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({'error': 'Missing fields'}), 400
    try:
        conn = mysql_helper.get_mysql_connection()
        cursor = conn.cursor(prepared=True)
        sql = "SELECT id, username, password FROM users WHERE email = %s"
        cursor.execute(sql, (email,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if not row:
            return jsonify({'error': 'Invalid credentials'}), 401
        user_id, username, hashed_str = row
        if isinstance(hashed_str, str):
            stored_hash = hashed_str.encode('utf-8')
        else:
            stored_hash = hashed_str
        if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return jsonify({'error': 'Invalid credentials'}), 401
        # Generate JWT token
        token = jwt_handler.create_token(user_id)
        resp = jsonify({'message': 'Login successful'})
        resp.set_cookie('token', token, httponly=True, samesite='Lax')
        return resp, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-auth', methods=['GET'])
@login_required
def check_auth():
    return '', 200

@app.route('/logout', methods=['GET'])
def logout():
    resp = redirect('/login-page')
    resp.set_cookie('token', '', expires=0, httponly=True, samesite='Lax')
    return resp

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/coach')
def coach():
    return render_template('coach.html')

@app.route('/vr')
def vr_session():
    global active_exercise
    env = request.args.get('env', 'gym') 
    active_exercise = request.args.get('ex', 'pushup')
    
    model_map = {
        'pushup': 'pushup_avatar',
        'squat': 'pistol_squads',
        'yoga': 'meditation',
        'cardio': 'Surya_namaskar'
    }
    model_filename = model_map.get(active_exercise, 'pushup_avatar')
    
    return render_template('index.html', env=env, exercise=active_exercise, model_filename=model_filename)

@socketio.on('start_workout')
def handle_start_workout(data):
    global is_workout_active, workout_mode, target_reps, start_time, target_duration, exercise_state
    
    workout_mode = data.get('mode', 'reps')
    is_workout_active = True
    
    # Reset the state dictionary for a fresh workout
    exercise_state = {'stage': 'up', 'reps': 0, 'lowest_angle': 180, 'status': 'Ready!'}
    
    if workout_mode == 'time':
        target_duration = int(data.get('target', 5)) * 60 
        start_time = time.time()
        emit('update_vr', {'status': 'Breathe...', 'score_text': 'Time Left: ...'})
    else:
        target_reps = int(data.get('target', 10))
        emit('update_vr', {'status': 'Get Ready!', 'score_text': f'Reps: 0 / {target_reps}'})

def process_webcam():
    global active_exercise, is_workout_active, exercise_state
    global target_reps, workout_mode, start_time, target_duration
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        score_text = ""
        current_status = "Waiting..."

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            if is_workout_active:
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # --- Send memory state to logic module ---
                    exercise_state, play_audio = exercise_logic.detect_exercise(
                        active_exercise, landmarks, exercise_state
                    )
                    
                    # Extract updated info
                    rep_count = exercise_state['reps']
                    current_status = exercise_state['status']
                    
                    if workout_mode == 'reps':
                        score_text = f"Reps: {rep_count} / {target_reps}"
                        if rep_count >= target_reps:
                            is_workout_active = False
                            current_status = "Goal Reached! Excellent Work."
                            socketio.emit('play_audio', {'file': 'good'}) 
                        
                        if play_audio and is_workout_active:
                            socketio.emit('play_audio', {'file': 'good'})
                            
                    elif workout_mode == 'time':
                        elapsed = time.time() - start_time
                        remaining = int(max(0, target_duration - elapsed))
                        mins, secs = divmod(remaining, 60)
                        score_text = f"Time Left: {mins}:{secs:02d}"
                        
                        if remaining <= 0:
                            is_workout_active = False
                            current_status = "Meditation Complete!"
                            socketio.emit('play_audio', {'file': 'good'})

                    # Only emit updates to VR
                    socketio.emit('update_vr', {
                        'score_text': score_text,
                        'status': current_status
                    })
                except Exception as e:
                    pass
            else:
                if workout_mode == 'time' and target_duration > 0 and (time.time() - start_time) >= target_duration:
                    current_status = "Session Done!"
                    score_text = "Time Left: 0:00"
                elif workout_mode == 'reps' and exercise_state['reps'] >= target_reps and target_reps > 0:
                    current_status = "Session Done!"
                    score_text = f"Reps: {target_reps} / {target_reps}"
                else:
                    current_status = "Click Start"
                    score_text = "Ready"
                
                socketio.emit('update_vr', {
                    'score_text': score_text,
                    'status': current_status
                })
        
        cv2.imshow('PhysioVR Vision', image) 
        if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

@socketio.on('send_chat')
def handle_chat(data):
    # ... (Keep your existing chat logic here) ...
    user_text = data['message']
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": "You are PhysioVR Coach. Keep answers under 50 words."},
                {"role": "user", "content": user_text}
            ]
        )
        ai_reply = response.choices[0].message.content
    except Exception as e:
        ai_reply = "I'm having trouble connecting to the cloud. Please check your internet."

    emit('receive_chat', {'role': 'ai', 'text': ai_reply})

if __name__ == '__main__':
    t = Thread(target=process_webcam)
    t.daemon = True
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)