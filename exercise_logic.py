import numpy as np
import mediapipe as mp
import random
import time
from collections import deque

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_robust_angle(landmarks, left_joints, right_joints, threshold=0.6):
    l_vis = all(landmarks[pt.value].visibility > threshold for pt in left_joints)
    r_vis = all(landmarks[pt.value].visibility > threshold for pt in right_joints)

    if l_vis and r_vis:
        l_coords = [[landmarks[pt.value].x, landmarks[pt.value].y] for pt in left_joints]
        r_coords = [[landmarks[pt.value].x, landmarks[pt.value].y] for pt in right_joints]
        return (calculate_angle(*l_coords) + calculate_angle(*r_coords)) / 2.0
    elif l_vis:
        l_coords = [[landmarks[pt.value].x, landmarks[pt.value].y] for pt in left_joints]
        return calculate_angle(*l_coords)
    elif r_vis:
        r_coords = [[landmarks[pt.value].x, landmarks[pt.value].y] for pt in right_joints]
        return calculate_angle(*r_coords)
    else:
        return None

def snapshot(landmarks):
    idxs = [11, 12, 23, 24, 25, 26]
    return {i: (landmarks[i].x, landmarks[i].y) for i in idxs}

def body_avg_velocity(landmarks, prev_snap, dt):
    if prev_snap is None or dt <= 0: return 0.0
    idxs = [11, 12, 23, 24, 25, 26]
    dists = [np.hypot(landmarks[i].x - prev_snap[i][0], landmarks[i].y - prev_snap[i][1]) for i in idxs]
    return float(np.mean(dists)) / dt

def detect_exercise(exercise_type, landmarks, state):
    play_audio = False
    current_time = time.time()
    
    # --- 1. INITIALIZE HISTORY BUFFERS IF EMPTY ---
    # (These were missing in the last version! They are required for the dashboard)
    if 'velocity_history' not in state: state['velocity_history'] = deque(maxlen=25)
    if 'hip_x_history' not in state: state['hip_x_history'] = deque(maxlen=30)
    if 'upward_vel_history' not in state: state['upward_vel_history'] = deque(maxlen=10)
    if 'knee_angle_buffer' not in state: state['knee_angle_buffer'] = deque(maxlen=10)
    if 'knee_time_buffer' not in state: state['knee_time_buffer'] = deque(maxlen=10)
    if 'calib_standing_hip_y' not in state: state['calib_standing_hip_y'] = None
    if 'consecutive_incomplete' not in state: state['consecutive_incomplete'] = 0
    if 'baseline_up_velocity' not in state: state['baseline_up_velocity'] = None
    if 'calib_hip_x_std' not in state: state['calib_hip_x_std'] = None
    if 'calib_up_velocities' not in state: state['calib_up_velocities'] = []
    if 'surya_transition_times' not in state: state['surya_transition_times'] = []
    if 'surya_last_transition_time' not in state: state['surya_last_transition_time'] = current_time
    if 'prev_lm_snap' not in state: state['prev_lm_snap'] = None
    if 'sn_pause_start' not in state: state['sn_pause_start'] = None
    if 'fatigue_index' not in state: state['fatigue_index'] = 0.0

    stage = state.get('stage', 'up')
    reps = state.get('reps', 0)
    lowest_angle = state.get('lowest_angle', 180)      
    worst_posture = state.get('worst_posture', 180)
    status = state.get('status', 'Get into position')
    
    fatigue_index = state['fatigue_index']
    fatigue_alerts = []
    
    prev_hip_y = state.get('prev_hip_y', None)
    prev_time = state.get('prev_time', current_time)
    dt = current_time - prev_time

    # ==========================================
    # 1. PUSH-UP LOGIC (Strict Form)
    # ==========================================
    if exercise_type == "pushup":
        elbow_angle = get_robust_angle(landmarks, 
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST))
        body_angle = get_robust_angle(landmarks,
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE))

        if elbow_angle is None or body_angle is None:
            state['status'] = "Camera blocked! Step back."
            return state, False

        if body_angle < 145:
            fatigue_alerts.append("Core Sagging")

        if stage == 'down' or elbow_angle < 130:
            lowest_angle = min(lowest_angle, elbow_angle)

        if elbow_angle > 160: 
            if stage == 'down':
                if body_angle < 150:
                    status = "Keep back straight! No core sag."
                    fatigue_alerts.append("Form Collapse")
                elif lowest_angle > 90:
                    status = "Go deeper! Chest to the floor."
                    fatigue_alerts.append("Incomplete Depth")
                else:
                    reps += 1
                    play_audio = True
                    status = "Textbook push-up!"
                
                stage = 'up'
                lowest_angle = 180
            elif stage == 'up' and "Textbook" not in status:
                if reps == 0: status = "Ready? Start pushing."
        
        elif elbow_angle < 85: 
            stage = 'down'
            status = "Hold... explode UP!"

    # ==========================================
    # 2. AIR SQUAT LOGIC (Strict Deadzones + Buffer History)
    # ==========================================
    elif exercise_type == "squat":
        knee_angle = get_robust_angle(landmarks,
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE))
        hip_angle = get_robust_angle(landmarks,
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE))
            
        l_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        r_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        hip_y_val = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y 
        hip_x_val = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2.0

        if knee_angle is None or hip_angle is None:
            state['status'] = "I can't see your full body! Step back."
            return state, False
            
        # 1. Update Continuous Buffers
        state['knee_angle_buffer'].append(knee_angle)
        state['knee_time_buffer'].append(current_time)
        state['hip_x_history'].append(hip_x_val)

        hip_delta = 0
        if prev_hip_y is not None:
            hip_delta = (hip_y_val - prev_hip_y) 
            vel_ = abs(hip_delta) / dt if dt > 0 else 0
            state['velocity_history'].append(vel_)
            if hip_delta < 0 and stage == "down": 
                state['upward_vel_history'].append(vel_)

        # 2. Downward Phase (Must break 120 to count as "down")
        if stage == 'down' or knee_angle < 120: 
            stage = 'down'
            lowest_angle = min(lowest_angle, knee_angle)
            worst_posture = min(worst_posture, hip_angle)
            status = "Drive up through your heels!"
            
        # 3. Upward Phase (Must stand fully straight > 160)
        if knee_angle >= 160: 
            if stage == 'down':
                # STRICT ACCURACY CHECKS
                if lowest_angle > 105: 
                    status = "Half-rep! Go lower next time."
                    fatigue_alerts.append("Incomplete Depth")
                elif worst_posture < 60:
                    status = "Keep your chest up!"
                    fatigue_alerts.append("Form Collapse")
                elif l_wrist_y > hip_y_val and r_wrist_y > hip_y_val:
                    status = "Raise your arms in front of you!"
                else:
                    reps += 1
                    play_audio = True
                    status = "Perfect air squat!"
                    
                    # Calibration Math Updates
                    if len(state['upward_vel_history']) > 0:
                        state['calib_up_velocities'].append(float(np.mean(state['upward_vel_history'])))
                        if len(state['calib_up_velocities']) >= 2:
                            state['baseline_up_velocity'] = float(np.mean(state['calib_up_velocities']))
                            if len(state['hip_x_history']) > 5:
                                state['calib_hip_x_std'] = float(np.std(state['hip_x_history']))

                state['upward_vel_history'].clear()
                lowest_angle = 180
                worst_posture = 180
            stage = 'up'
            if reps == 0: status = "Ready for squats."

        if knee_angle > 155 and state['calib_standing_hip_y'] is None:
            state['calib_standing_hip_y'] = hip_y_val

        # --- 4. CONTINUOUS FATIGUE CHECKS ---
        c1_fired = c2_fired = c3_fired = c4_fired = c5_fired = False

        if len(state['knee_angle_buffer']) >= 2:
            for i in range(len(state['knee_time_buffer'])):
                if (current_time - state['knee_time_buffer'][i]) <= 0.2:
                    if (state['knee_angle_buffer'][i] - knee_angle) > 25:
                        c1_fired = True
                    break

        if hip_delta > 0.04 and stage == "down": c2_fired = True

        if state['baseline_up_velocity'] is not None and stage == "down" and len(state['upward_vel_history']) >= 3:
            rv = float(np.mean(list(state['upward_vel_history'])[-3:]))
            if rv < state['baseline_up_velocity'] * 0.4:
                c3_fired = True

        if state['calib_standing_hip_y'] is not None and stage == "up" and knee_angle > 150:
            thr = state['calib_standing_hip_y'] + (state['calib_standing_hip_y'] * 0.1)
            if hip_y_val > thr: state['consecutive_incomplete'] += 1
            else: state['consecutive_incomplete'] = 0
            if state['consecutive_incomplete'] >= 10: c4_fired = True

        if state['calib_hip_x_std'] is not None and state['calib_hip_x_std'] > 0.01 and len(state['hip_x_history']) >= 30:
            if float(np.std(state['hip_x_history'])) > state['calib_hip_x_std'] * 2.5:
                c5_fired = True

        if c1_fired: fatigue_alerts.append("Knee Collapse")
        if c2_fired: fatigue_alerts.append("Hip Drop")
        if c3_fired: fatigue_alerts.append("Speed Reduced")
        if c4_fired: fatigue_alerts.append("Incomplete Ext")
        if c5_fired: fatigue_alerts.append("Balance Unstable")
            
        state['prev_hip_y'] = hip_y_val


    # ==========================================
    # 3. SURYA NAMASKAR LOGIC
    # ==========================================
    elif exercise_type == "cardio":
        knee_angle = get_robust_angle(landmarks,
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE))
            
        avg_sh_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2.0
        avg_hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2.0
        avg_ank_y = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2.0
        avg_wr_y = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y) / 2.0

        if knee_angle is None:
            state['status'] = "Camera blocked! Need full body view."
            return state, False

        vel = body_avg_velocity(landmarks, state['prev_lm_snap'], dt)
        state['prev_lm_snap'] = snapshot(landmarks)

        # Rep Tracking
        if avg_wr_y < avg_sh_y and knee_angle > 150:
            if stage == 'down':
                gap = current_time - state['surya_last_transition_time']
                if gap > 0.5:
                    state['surya_transition_times'].append(gap)
                    if len(state['surya_transition_times']) >= 2:
                        bt = float(np.mean(state['surya_transition_times']))
                        if gap > bt * 1.5:
                            fatigue_alerts.append("Cycle Slowing")

                reps += 1
                play_audio = True
                status = "Great flow! Cycle complete."
                state['surya_last_transition_time'] = current_time
            stage = 'up'
        elif avg_wr_y > avg_hip_y and avg_sh_y > avg_hip_y - 0.2:
            stage = 'down'

        # Fatigue Checks
        is_plank = abs(avg_sh_y - avg_ank_y) < 0.15 and knee_angle > 140
        if is_plank:
            expected_hip_y = (avg_sh_y + avg_ank_y) / 2.0
            if avg_hip_y > expected_hip_y + 0.05: 
                fatigue_alerts.append("Plank Hip Sag")
            if knee_angle < 130: 
                fatigue_alerts.append("Knee Drop")

        if vel < 0.02: 
            if state['sn_pause_start'] is None:
                state['sn_pause_start'] = current_time
            elif current_time - state['sn_pause_start'] > 4.0: 
                fatigue_alerts.append("Abnormal Pause")
        else:
            state['sn_pause_start'] = None


    # ==========================================
    # 4. YOGA / MEDITATION LOGIC
    # ==========================================
    elif exercise_type == "yoga":
        status = "Focus on your breathing..."

    # ==========================================
    # FATIGUE SCORING MATH
    # ==========================================
    n_fired = len(fatigue_alerts)
    if n_fired >= 2: fatigue_index = min(100.0, fatigue_index + (8.0 * n_fired))
    elif n_fired == 1: fatigue_index = min(100.0, fatigue_index + 2.0)
    else: fatigue_index = max(0.0, fatigue_index - 0.2) 

    # Save to state
    state['stage'] = stage
    state['reps'] = reps
    state['lowest_angle'] = lowest_angle
    state['worst_posture'] = worst_posture
    state['status'] = status
    state['fatigue_index'] = fatigue_index
    state['fatigue_alerts'] = fatigue_alerts
    state['prev_time'] = current_time

    return state, play_audio