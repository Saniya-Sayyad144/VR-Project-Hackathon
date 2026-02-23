import os
import cv2
import mediapipe as mp
import numpy as np
import time
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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "dev_secret")
socketio = SocketIO(app, cors_allowed_origins="*")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_pose    = mp.solutions.pose
pose       = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

POSE_DRAW_SPEC = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=3)
POSE_CONN_SPEC = mp_drawing.DrawingSpec(color=(160, 160, 160), thickness=2)

# --- 3-STATE SYSTEM TO FIX REDIRECT BUG ---
session_status = 'idle' # Can be 'idle', 'active', or 'completed'
active_exercise = "pushup"
workout_mode = 'reps'  
target_reps = 10
start_time = 0
target_duration = 0 
exercise_state = {}

# --- FRIEND'S GLOBAL COGNITIVE VARIABLES ---
EAR_THRESHOLD      = 0.23
BLINK_MIN_DURATION = 0.1
BLINK_MAX_DURATION = 0.4
NOSE_INDEX         = 1
CHIN_INDEX         = 152
LEFT_EYE           = [33, 160, 158, 133, 153, 144]
RIGHT_EYE          = [362, 385, 387, 263, 373, 380]
HEAD_DROP_COG      = 70

class C:
    BG_DARK        = (30,  30,  30)    
    BG_PANEL       = (245, 245, 245)   
    BG_HEADER      = (255, 255, 255)   
    BG_STATUS_OK   = (235, 252, 235)   
    BG_STATUS_WARN = (255, 243, 224)   
    BG_STATUS_CRIT = (255, 230, 230)   
    TEXT_DARK      = (30,  30,  30)    
    TEXT_MED       = (80,  80,  80)    
    TEXT_ACCENT    = (0,   120, 200)   
    OK             = (34,  139, 34)    
    WARN           = (0,   140, 255)   
    CRIT           = (30,  30,  200)   
    BAR_OK         = (60,  179, 60)
    BAR_WARN       = (0,   160, 255)
    BAR_CRIT       = (40,  40,  210)
    BAR_BG         = (210, 210, 210)   
    BAR_BORDER     = (160, 160, 160)
    PANEL_BORDER   = (180, 180, 180)
    DIVIDER        = (200, 200, 200)
    HEADER_LINE    = (0,   120, 200)   

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX

# --- FRIEND'S UI DRAWING FUNCTIONS ---
def txt(frame, text, x, y, color=C.TEXT_DARK, scale=0.52, thick=1, font=FONT):
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
def panel(frame, x1, y1, x2, y2, bg=C.BG_PANEL, border=C.PANEL_BORDER):
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)
def divider(frame, x1, x2, y, color=C.DIVIDER):
    cv2.line(frame, (x1, y), (x2, y), color, 1, cv2.LINE_AA)
def clinical_bar(frame, value, max_val, x, y, bar_w, bar_h=14, label="", unit="", low_good=True):
    ratio  = max(0.0, min(1.0, value / max_val))
    filled = int(ratio * bar_w)
    col = (C.BAR_OK if ratio < 0.35 else C.BAR_WARN if ratio < 0.70 else C.BAR_CRIT) if low_good else (C.BAR_CRIT if ratio < 0.35 else C.BAR_WARN if ratio < 0.70 else C.BAR_OK)
    if label: txt(frame, label, x, y - 3, C.TEXT_MED, scale=0.42)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), C.BAR_BG, -1)
    if filled > 0: cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), col, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), C.BAR_BORDER, 1)
    txt(frame, f"{value:.1f}{unit}", x + bar_w + 6, y + bar_h - 1, C.TEXT_MED, scale=0.42)
def status_chip(frame, text, x, y, w_chip, level="ok"):
    h_chip = 22
    bg  = {"ok": C.BG_STATUS_OK, "warn": C.BG_STATUS_WARN, "crit": C.BG_STATUS_CRIT}[level]
    col = {"ok": C.OK, "warn": C.WARN, "crit": C.CRIT}[level]
    cv2.rectangle(frame, (x, y), (x + w_chip, y + h_chip), bg, -1)
    cv2.rectangle(frame, (x, y), (x + w_chip, y + h_chip), col, 1)
    ts = cv2.getTextSize(text, FONT, 0.44, 1)[0]
    txt(frame, text, x + (w_chip - ts[0]) // 2, y + (h_chip + ts[1]) // 2 - 1, col, scale=0.44, thick=1)
def draw_header(canvas, W, mode_label, exercise_label=""):
    H_HDR = 52
    cv2.rectangle(canvas, (0, 0), (W, H_HDR), C.BG_HEADER, -1)
    cv2.line(canvas, (0, H_HDR), (W, H_HDR), C.HEADER_LINE, 2)
    txt(canvas, "VR Health Monitor", 14, 22, C.TEXT_ACCENT, scale=0.65, thick=2)
    txt(canvas, "Remote Fatigue Detection System", 14, 42, C.TEXT_MED, scale=0.40)
    badge = mode_label + (f"  |  {exercise_label}" if exercise_label else "")
    ts = cv2.getTextSize(badge, FONT, 0.58, 2)[0]
    txt(canvas, badge, (W - ts[0]) // 2, 32, C.TEXT_ACCENT, scale=0.58, thick=2)
def draw_bottom_bar(canvas, W, H, status_text, status_level, keys_hint):
    BAR_H = 48
    y0 = H - BAR_H
    bg = {"ok": C.BG_STATUS_OK, "warn": C.BG_STATUS_WARN, "crit": C.BG_STATUS_CRIT}.get(status_level, C.BG_PANEL)
    col = {"ok": C.OK, "warn": C.WARN, "crit": C.CRIT}.get(status_level, C.TEXT_DARK)
    cv2.rectangle(canvas, (0, y0), (W, H), bg, -1)
    cv2.line(canvas, (0, y0), (W, y0), C.PANEL_BORDER, 1)
    txt(canvas, status_text, 20, y0 + 30, col, scale=0.75, thick=2)
def draw_cognitive_panel(canvas, px, py, pw, ear, blinks_per_min, fatigue_idx, head_status, head_level):
    panel(canvas, px, py, px + pw, py + 230)
    txt(canvas, "COGNITIVE MODE", px + 12, py + 20, C.TEXT_ACCENT, scale=0.52, thick=2)
    divider(canvas, px + 12, px + pw - 12, py + 28)
    txt(canvas, "Eye Aspect Ratio (EAR)", px + 12, py + 50, C.TEXT_MED, scale=0.42)
    txt(canvas, f"{ear:.3f}", px + pw - 60, py + 50, C.CRIT if ear < EAR_THRESHOLD else C.OK, scale=0.50, thick=2)
    txt(canvas, "Blink Rate", px + 12, py + 76, C.TEXT_MED, scale=0.42)
    txt(canvas, f"{int(blinks_per_min)} / min", px + pw - 80, py + 76, C.CRIT if blinks_per_min < 8 else C.OK, scale=0.50, thick=2)
    divider(canvas, px + 12, px + pw - 12, py + 88)
    txt(canvas, "Fatigue Index", px + 12, py + 108, C.TEXT_MED, scale=0.42)
    fi_pct = (fatigue_idx / 5.0) * 100.0
    txt(canvas, f"{fi_pct:.0f}%", px + pw - 56, py + 108, C.OK if fi_pct < 35 else C.WARN if fi_pct < 70 else C.CRIT, scale=0.50, thick=2)
    clinical_bar(canvas, fatigue_idx, 5.0, px + 12, py + 118, bar_w=pw - 54, bar_h=14, low_good=True)
    divider(canvas, px + 12, px + pw - 12, py + 144)
    txt(canvas, "Head Position", px + 12, py + 166, C.TEXT_MED, scale=0.42)
    status_chip(canvas, head_status, px + 12, py + 174, pw - 24, level=head_level)
def draw_exercise_panel(canvas, px, py, pw, fi_pct, status_msgs, ex_status, ex_level, exercise_type):
    ph = max(280, 52 + 42 + (len(status_msgs) + 1) * 24 + 50)
    panel(canvas, px, py, px + pw, py + ph)
    txt(canvas, exercise_type + " MODE", px + 12, py + 20, C.TEXT_ACCENT, scale=0.52, thick=2)
    divider(canvas, px + 12, px + pw - 12, py + 28)
    
    txt(canvas, "Fatigue Index", px + 12, py + 50, C.TEXT_MED, scale=0.42)
    txt(canvas, f"{fi_pct:.0f}%", px + pw - 56, py + 50, C.OK if fi_pct < 31 else C.WARN if fi_pct < 61 else C.CRIT, scale=0.55, thick=2)
    clinical_bar(canvas, fi_pct, 100.0, px + 12, py + 60, bar_w=pw - 54, bar_h=14, low_good=True)
    status_chip(canvas, "Normal" if fi_pct <= 30 else "Mild Fatigue" if fi_pct <= 60 else "Severe Fatigue", px + 12, py + 82, pw - 24, level="ok" if fi_pct <= 30 else "warn" if fi_pct <= 60 else "crit")
    divider(canvas, px + 12, px + pw - 12, py + 114)
    txt(canvas, "Active Signals", px + 12, py + 132, C.TEXT_MED, scale=0.42)
    row_y = py + 148
    if status_msgs:
        for msg in status_msgs:
            is_alert = any(kw in msg for kw in ["Collapse", "Drop", "Reduced", "Unstable", "Incomplete", "Slowing", "Sag", "Pause", "Slow", "Raise", "Dropping"])
            status_chip(canvas, msg, px + 12, row_y, pw - 24, level="crit" if is_alert else "ok")
            row_y += 26
    else:
        status_chip(canvas, "Monitoring...", px + 12, row_y, pw - 24, level="ok")
        row_y += 26
    divider(canvas, px + 12, px + pw - 12, row_y + 6)
    status_chip(canvas, ex_status, px + 12, row_y + 14, pw - 24, level=ex_level)
def draw_alert_banner(canvas, W, text, alpha=0.72):
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 52), (W, 52 + 46), (40, 40, 200), -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    ts  = cv2.getTextSize(text, FONT, 0.85, 2)[0]
    txt(canvas, text, (W - ts[0]) // 2, 52 + 23 + ts[1] // 2, (255, 255, 255), scale=0.85, thick=2)

def eye_aspect_ratio(pts):
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C_ = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C_)

# --- DB ROUTES ---
def save_session(user_id, exercise, reps, duration, fatigue):
    try:
        conn = mysql_helper.get_mysql_connection()
        cursor = conn.cursor()
        sql = "INSERT INTO sessions (user_id, exercise_name, reps, duration, fatigue, vr_time) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (user_id, exercise, reps, duration, fatigue, duration))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e: print("Session save error:", e)

@socketio.on('end_workout')
def handle_end_workout(data):
    global exercise_state, active_exercise, start_time
    user_id = data.get('user_id')
    reps = data.get('reps', exercise_state.get('reps', 0))
    fatigue = data.get('fatigue', exercise_state.get('fatigue_index', 0.0))
    duration = int(time.time() - start_time)
    save_session(user_id, active_exercise, reps, duration, fatigue)
    emit('workout_saved', {'success': True})

@app.route('/api/save-session', methods=['POST'])
@login_required
def api_save_session():
    user_id = request.user_id
    data = request.get_json()
    save_session(user_id, data.get('exercise', 'pushup'), int(data.get('reps', 0)), int(data.get('duration', 0)), float(data.get('fatigue', 0.0)))
    return jsonify({'success': True})

@app.route('/')
def home(): return render_template('home.html', auth_check=True)
@app.route('/login-page', methods=['GET'])
def login_page(): return render_template('login.html')
@app.route('/register-page', methods=['GET'])
def register_page(): return render_template('register.html')
@app.route('/dashboard')
@login_required
def dashboard(): return render_template('dashboard.html')
@app.route('/coach')
def coach(): return render_template('coach.html')
@app.route('/about')
def about(): return render_template('about.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data.get('username') or not data.get('email') or not data.get('password'): return jsonify({'error': 'Missing fields'}), 400
    hashed_str = bcrypt.hashpw(data.get('password').encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        conn = mysql_helper.get_mysql_connection()
        cursor = conn.cursor(prepared=True)
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (data.get('username'), data.get('email'), hashed_str))
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    try:
        conn = mysql_helper.get_mysql_connection()
        cursor = conn.cursor(prepared=True)
        cursor.execute("SELECT id, username, password FROM users WHERE email = %s", (data.get('email'),))
        row = cursor.fetchone()
        if not row or not bcrypt.checkpw(data.get('password').encode('utf-8'), row[2].encode('utf-8') if isinstance(row[2], str) else row[2]):
            return jsonify({'error': 'Invalid credentials'}), 401
        resp = jsonify({'message': 'Login successful'})
        resp.set_cookie('token', jwt_handler.create_token(row[0]), httponly=True, samesite='Lax')
        return resp, 200
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/check-auth', methods=['GET'])
@login_required
def check_auth(): return '', 200

@app.route('/logout', methods=['GET'])
def logout():
    resp = redirect('/login-page')
    resp.set_cookie('token', '', expires=0, httponly=True, samesite='Lax')
    return resp

@app.route('/api/dashboard-data', methods=['GET'])
@login_required
def api_dashboard_data():
    try:
        conn = mysql_helper.get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COALESCE(SUM(duration),0) AS total_exercise_time, COALESCE(SUM(vr_time),0) AS total_vr_time, COALESCE(SUM(reps),0) AS total_reps, AVG(fatigue) AS avg_fatigue FROM sessions WHERE user_id = %s", (request.user_id,))
        agg = cursor.fetchone() or {'total_exercise_time': 0, 'total_vr_time': 0, 'total_reps': 0, 'avg_fatigue': None}
        cursor.execute("SELECT DATE(created_at) as date, exercise_name, duration, reps, fatigue FROM sessions WHERE user_id = %s ORDER BY created_at DESC LIMIT 100", (request.user_id,))
        sessions = [{'date': r.get('date').strftime('%Y-%m-%d') if r.get('date') else None, 'exercise_name': r.get('exercise_name'), 'duration': int(r.get('duration') or 0), 'reps': int(r.get('reps') or 0), 'fatigue': r.get('fatigue')} for r in cursor.fetchall()]
        return jsonify({'total_exercise_time': int(agg.get('total_exercise_time') or 0), 'total_vr_time': int(agg.get('total_vr_time') or 0), 'total_reps': int(agg.get('total_reps') or 0), 'avg_fatigue': float(agg['avg_fatigue']) if agg.get('avg_fatigue') is not None else None, 'sessions': sessions})
    except: return jsonify({'total_exercise_time': 0, 'total_vr_time': 0, 'total_reps': 0, 'avg_fatigue': None, 'sessions': []})

@app.route('/api/ai/day-analysis', methods=['POST'])
@login_required
def api_ai_day_analysis():
    try:
        from datetime import datetime
        conn = mysql_helper.get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        today = datetime.now().date()
        cursor.execute("SELECT exercise_name, duration, reps, fatigue FROM sessions WHERE user_id = %s AND DATE(created_at) = %s ORDER BY created_at", (request.user_id, today))
        sessions = cursor.fetchall()
        if not sessions:
            return jsonify({'analysis': 'No sessions recorded today. Start your first session to get AI feedback!'}), 200
        sessions_text = '\n'.join([f"- {s['exercise_name']}: {s['duration']}s, {s['reps']} reps, Fatigue: {s['fatigue']}" for s in sessions])
        prompt = f"Analyze this user's workout session for today and provide motivational, constructive feedback. Keep it concise (2-3 sentences).\n\nToday's Sessions:\n{sessions_text}\n\nProvide insights on their performance, energy levels (fatigue), and suggestions for improvement."
        response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}], max_tokens=300)
        analysis = response.choices[0].message.content
        return jsonify({'analysis': analysis}), 200
    except Exception as e:
        return jsonify({'error': f'AI analysis unavailable: {str(e)}'}), 500

@app.route('/vr')
def vr_session():
    global active_exercise
    env = request.args.get('env', 'gym') 
    active_exercise = request.args.get('ex', 'pushup')
    model_filename = {'pushup': 'pushup_avatar', 'squat': 'pistol_squads', 'yoga': 'meditation', 'cardio': 'Surya_namaskar'}.get(active_exercise, 'pushup_avatar')
    return render_template('index.html', env=env, exercise=active_exercise, model_filename=model_filename)

@socketio.on('start_workout')
def handle_start_workout(data):
    global session_status, workout_mode, target_reps, start_time, target_duration, exercise_state
    
    workout_mode = data.get('mode', 'reps')
    
    # HARD RESET MEMORY
    exercise_state.clear()
    exercise_state['reps'] = 0
    exercise_state['stage'] = 'up'
    exercise_state['lowest_angle'] = 180
    exercise_state['worst_posture'] = 180
    exercise_state['fatigue_index'] = 0.0
    exercise_state['fatigue_alerts'] = []
    
    session_status = 'active'
    
    if workout_mode == 'time':
        target_duration = int(data.get('target', 5)) * 60 
        start_time = time.time()
        emit('update_vr', {'status': 'Breathe...', 'score_text': 'Time Left: ...'})
    else:
        target_reps = int(data.get('target', 10))
        emit('update_vr', {'status': 'Get Ready!', 'score_text': f'Reps: 0 / {target_reps}'})

# --- UNIFIED VISION ENGINE ---
def process_webcam():
    global active_exercise, session_status, workout_mode, target_reps, start_time, target_duration, exercise_state
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    WIN_NAME = "VR Health Monitor - Remote Dashboard"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    last_active_exercise = None
    alert_text = ""
    alert_until = 0.0
    smoothed_fatigue = 0.0
    
    closed_start_time = None
    blink_start = None
    blink_timestamps = []
    fatigue_index_cog = 0.0
    blink_rate_smoothed = 15.0

    while True:
        ret, frame = cap.read()
        if not ret: continue
        current_time = time.time()
        fh, fw, _ = frame.shape

        CW, CH = 1280, 720
        HDR_H, BOT_H, PANEL_W, PAD = 52, 48, 290, 10
        CAM_X, CAM_Y = PAD, HDR_H + PAD
        CAM_W, CAM_H = CW - PANEL_W - PAD * 3, CH - HDR_H - BOT_H - PAD * 2
        PNL_X, PNL_Y = CW - PANEL_W - PAD, HDR_H + PAD

        canvas = np.full((CH, CW, 3), 240, dtype=np.uint8)
        cam_resized = cv2.resize(frame, (CAM_W, CAM_H))
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb)

        if active_exercise in ["squat", "pushup", "cardio"]:
            mode = "EXERCISE"
            sys_ex_type = "SQUATS" if active_exercise == "squat" else "SURYA NAMASKAR" if active_exercise == "cardio" else "PUSHUPS"
        else:
            mode = "COGNITIVE"
            sys_ex_type = "YOGA / MEDITATION"
            
        if active_exercise != last_active_exercise:
            exercise_state.clear()
            last_active_exercise = active_exercise
            fatigue_index_cog = 0.0
            smoothed_fatigue = 0.0
            blink_timestamps.clear()

        head_drop = False
        ex_main_status = "Waiting for VR..."
        ex_main_level = "warn"
        bottom_status = "System Ready"
        bottom_level = "ok"
        vr_status_msg = "Tracking..."
        vr_fatigue_alert = ""

        # --- EXERCISE MODE ---
        if mode == "EXERCISE":
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(cam_resized, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, POSE_DRAW_SPEC, POSE_CONN_SPEC)
            
            if pose_results.pose_landmarks and session_status == 'active':
                lm = pose_results.pose_landmarks.landmark
                exercise_state, play_audio = exercise_logic.detect_exercise(active_exercise, lm, exercise_state)
                
                rep_count   = exercise_state.get('reps', 0)
                status_msgs = exercise_state.get('fatigue_alerts', [])
                raw_f_index = exercise_state.get('fatigue_index', 0.0)
                vr_status_msg = exercise_state.get('status', 'Tracking...')
                
                # Apply smoothing to prevent jumping
                smoothed_fatigue = (0.1 * raw_f_index) + (0.9 * smoothed_fatigue)

                if len(status_msgs) >= 2 or smoothed_fatigue > 75:
                    ex_main_status = "FATIGUE DETECTED"
                    ex_main_level = "crit"
                    alert_text = "⚠ FATIGUE DETECTED"
                    alert_until = current_time + 2.5
                    bottom_status = "FATIGUE DETECTED — STOP!"
                    bottom_level = "crit"
                    vr_fatigue_alert = "⚠ " + (status_msgs[-1] if status_msgs else "Fatigue")
                elif smoothed_fatigue > 60:
                    ex_main_status = "Moderate Fatigue"
                    ex_main_level = "crit"
                    bottom_status = "Moderate Fatigue"
                    bottom_level = "warn"
                elif smoothed_fatigue > 30:
                    ex_main_status = "Mild Fatigue"
                    ex_main_level = "warn"
                    bottom_status = "Mild Fatigue"
                    bottom_level = "warn"
                else:
                    ex_main_status = "Exercising Normally"
                    ex_main_level = "ok"
                    bottom_status = "Tracking OK"
                    bottom_level = "ok"

                draw_exercise_panel(canvas, PNL_X, PNL_Y, PANEL_W, smoothed_fatigue, status_msgs, ex_main_status, ex_main_level, sys_ex_type)
                cv2.putText(cam_resized, f"REPS: {rep_count}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 140, 255), 3, cv2.LINE_AA)
            
            else:
                draw_exercise_panel(canvas, PNL_X, PNL_Y, PANEL_W, 0.0, [], "Waiting for VR Start", "warn", sys_ex_type)
                cv2.putText(cam_resized, "REPS: 0", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 140, 255), 3, cv2.LINE_AA)
                cv2.putText(cam_resized, "WAITING TO START IN VR...", (30, 110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # --- COGNITIVE MODE ---
        if face_results.multi_face_landmarks:
            for fl in face_results.multi_face_landmarks:
                nose_y = int(fl.landmark[NOSE_INDEX].y * fh)
                chin_y = int(fl.landmark[CHIN_INDEX].y * fh)
                if (chin_y - nose_y) < HEAD_DROP_COG: head_drop = True

                if mode == "COGNITIVE":
                    lep = [(int(fl.landmark[i].x*fw), int(fl.landmark[i].y*fh)) for i in LEFT_EYE]
                    rep_pts = [(int(fl.landmark[i].x*fw), int(fl.landmark[i].y*fh)) for i in RIGHT_EYE]
                    ear = (eye_aspect_ratio(lep) + eye_aspect_ratio(rep_pts)) / 2.0

                    if ear < EAR_THRESHOLD:
                        if blink_start is None: blink_start = current_time
                        if closed_start_time is None: closed_start_time = current_time
                    else:
                        if blink_start is not None and BLINK_MIN_DURATION < (current_time - blink_start) < BLINK_MAX_DURATION:
                            blink_timestamps.append(current_time)
                            blink_start = None
                        closed_start_time = None

                    blink_timestamps = [t for t in blink_timestamps if current_time - t <= 60]
                    blink_rate_smoothed = (0.1 * len(blink_timestamps) + 0.9 * blink_rate_smoothed)

                    fi_inc = 0.0
                    if closed_start_time and current_time - closed_start_time > 2: fi_inc += 0.05
                    if head_drop: fi_inc += 0.02
                    if blink_rate_smoothed < 8: fi_inc += 0.01

                    if fi_inc > 0: fatigue_index_cog = min(5.0, fatigue_index_cog + fi_inc)
                    else: fatigue_index_cog = max(0.0, fatigue_index_cog - 0.01)

                    if fatigue_index_cog >= 3:
                        bottom_status = "HIGH FATIGUE ALERT!"
                        bottom_level = "crit"
                        vr_fatigue_alert = "⚠ Wake Up!"
                    elif fatigue_index_cog >= 1:
                        bottom_status = "MODERATE FATIGUE"
                        bottom_level = "warn"
                    else:
                        bottom_status = "RELAXED"
                        bottom_level = "ok"

                    draw_cognitive_panel(canvas, PNL_X, PNL_Y, PANEL_W, ear, blink_rate_smoothed, fatigue_index_cog, "Head Drop!" if head_drop else "Normal", "crit" if head_drop else "ok")

        canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = cam_resized
        draw_header(canvas, CW, mode + " MODE", sys_ex_type if mode=="EXERCISE" else "")
        if mode == "EXERCISE" and current_time < alert_until and alert_text: draw_alert_banner(canvas, CW, alert_text)
        draw_bottom_bar(canvas, CW, CH, bottom_status, bottom_level, "VR Link Active")
        cv2.rectangle(canvas, (CAM_X-1, CAM_Y-1), (CAM_X+CAM_W, CAM_Y+CAM_H), C.PANEL_BORDER, 1)

        # --- FIX: THE 3-STATE VR SOCKET LOGIC ---
        if session_status == 'active':
            score_text = ""
            if workout_mode == 'reps':
                current_reps = exercise_state.get('reps', 0)
                score_text = f"Reps: {current_reps} / {target_reps}"
                
                if current_reps >= target_reps:
                    session_status = 'completed'
                    vr_status_msg = "Goal Reached! Exiting..."
                    socketio.emit('play_audio', {'file': 'good'})
            elif workout_mode == 'time':
                elapsed = time.time() - start_time
                remaining = int(max(0, target_duration - elapsed))
                mins, secs = divmod(remaining, 60)
                score_text = f"Time Left: {mins}:{secs:02d}"
                
                if remaining <= 0:
                    session_status = 'completed'
                    vr_status_msg = "Meditation Complete!"
                    socketio.emit('play_audio', {'file': 'good'})

            socketio.emit('update_vr', {
                'score_text': score_text,
                'status': vr_status_msg,
                'fatigue_score': smoothed_fatigue if mode=="EXERCISE" else (fatigue_index_cog/5.0)*100, 
                'fatigue_text': vr_fatigue_alert,
                'reps': exercise_state.get('reps', 0)
            })

        elif session_status == 'completed':
            # Lock the UI into the "Session Done" state until VR redirects
            score_text = f"Reps: {target_reps}/{target_reps}" if workout_mode == 'reps' else "Time Left: 0:00"
            socketio.emit('update_vr', {
                'status': 'Session Done!', 
                'score_text': score_text,
                'fatigue_score': 0.0,
                'fatigue_text': '',
                'reps': target_reps if workout_mode == 'reps' else 0
            })
            
        else: # 'idle'
            socketio.emit('update_vr', {
                'status': 'Click Start', 
                'score_text': 'Ready',
                'fatigue_score': 0.0,
                'fatigue_text': '',
                'reps': 0
            })
        
        cv2.imshow(WIN_NAME, canvas) 
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

@socketio.on('send_chat')
def handle_chat(data):
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