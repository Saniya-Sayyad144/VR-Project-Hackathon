import numpy as np
import mediapipe as mp
import random
import time

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_robust_angle(landmarks, left_joints, right_joints, threshold=0.6):
    """Used for two-sided exercises like pushups."""
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

def get_single_angle(landmarks, joints, threshold=0.5):
    """NEW: Gets angle for a single side if visible. Crucial for unilateral moves like Pistol Squats."""
    if all(landmarks[pt.value].visibility > threshold for pt in joints):
        coords = [[landmarks[pt.value].x, landmarks[pt.value].y] for pt in joints]
        return calculate_angle(*coords)
    return None

def detect_exercise(exercise_type, landmarks, state):
    """Routes the landmark data to the correct exercise logic using state memory."""
    play_audio = False
    current_time = time.time()
    
    # 1. Read current state memory
    stage = state.get('stage', 'up')
    reps = state.get('reps', 0)
    lowest_angle = state.get('lowest_angle', 180)      
    worst_posture = state.get('worst_posture', 180)    
    status = state.get('status', 'Get into position')
    
    # 2. NEW: Fatigue Engine Variables
    fatigue_index = state.get('fatigue_index', 0.0)
    fatigue_alerts = state.get('fatigue_alerts', [])
    prev_hip_y = state.get('prev_hip_y', None)
    prev_time = state.get('prev_time', current_time)
    baseline_vel = state.get('baseline_vel', None)

    # ==========================================
    # 1. PUSH-UP LOGIC (Unchanged)
    # ==========================================
    if exercise_type == "pushup":
        elbow_angle = get_robust_angle(landmarks, 
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST))
        body_angle = get_robust_angle(landmarks,
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE))

        if elbow_angle is None or body_angle is None:
            state['status'] = "Camera blocked! Step back so I can see you."
            return state, False

        if stage == 'down' or elbow_angle < 130:
            lowest_angle = min(lowest_angle, elbow_angle)

        if elbow_angle > 160: 
            if stage == 'down':
                if body_angle < 150:
                    status = random.choice(["Keep your core tight! Don't drop your hips!", "Back straight! Squeeze your abs!"])
                elif lowest_angle > 90:
                    status = random.choice(["Come on, your chest isn't going down! Deeper!", "No half reps! Show more effort!"])
                else:
                    reps += 1
                    play_audio = True
                    status = random.choice(["Textbook push-up! Great job.", "Perfect rep, keep pushing!", "That's it! Nice depth."])
                
                stage = 'up'
                lowest_angle = 180
            elif stage == 'up' and status not in ["Textbook push-up! Great job.", "Perfect rep, keep pushing!", "That's it! Nice depth."]:
                if reps == 0: status = "Ready? Let's see those push-ups."

        elif elbow_angle < 85: 
            stage = 'down'
            status = "Hold it... Now explode UP!"
        elif 85 <= elbow_angle <= 160:
            if stage == 'up': status = "Control the descent..."


    # ==========================================
    # 2. PISTOL SQUAT LOGIC (With Fatigue Engine)
    # ==========================================
    elif exercise_type == "squat":
        # Calculate left and right legs independently
        l_knee = get_single_angle(landmarks, (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE))
        r_knee = get_single_angle(landmarks, (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE))
        
        l_hip = get_single_angle(landmarks, (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE))
        r_hip = get_single_angle(landmarks, (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE))

        # Dynamically find the "Active" leg
        active_knee = None
        active_hip = None
        
        if l_knee and r_knee:
            if l_knee < r_knee:
                active_knee, active_hip = l_knee, l_hip
            else:
                active_knee, active_hip = r_knee, r_hip
        elif l_knee:
            active_knee, active_hip = l_knee, l_hip
        elif r_knee:
            active_knee, active_hip = r_knee, r_hip
            
        if active_knee is None or active_hip is None:
            state['status'] = "I can't see your legs! Step back."
            return state, False
            
        # --- FATIGUE: Calculate Hip Velocity ---
        hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2.0
        dt = current_time - prev_time
        current_vel = 0
        if prev_hip_y is not None and dt > 0:
            current_vel = abs(hip_y - prev_hip_y) / dt

        # Reset active alerts for this frame
        fatigue_alerts.clear()

        # Memory: Record the lowest point of the single leg
        if stage == 'down' or active_knee < 140:
            lowest_angle = min(lowest_angle, active_knee)
            worst_posture = min(worst_posture, active_hip)
            
        # Transitions & Fatigue Checks
        if active_knee > 160: 
            if stage == 'down':
                # 1. Check Depth (Incomplete Extension Fatigue)
                if lowest_angle > 110: 
                    status = random.choice(["Go lower! Get that balance.", "Half-rep! Try to sink deeper on that single leg."])
                    fatigue_alerts.append("Incomplete Depth")
                    fatigue_index += 15.0
                # 2. Check Form (Core Collapse Fatigue)
                elif worst_posture < 60:
                    status = random.choice(["Keep your chest up!", "Don't collapse forward! Engage your core."])
                    fatigue_alerts.append("Form Collapse")
                    fatigue_index += 20.0
                else:
                    # Good rep! Check upward velocity fatigue
                    if baseline_vel is None:
                        baseline_vel = current_vel # Calibrate on first good rep
                    elif current_vel < (baseline_vel * 0.5): # 50% slower than normal
                        fatigue_alerts.append("Speed Dropping")
                        fatigue_index += 10.0

                    reps += 1
                    play_audio = True
                    status = random.choice(["Amazing balance! Great pistol squat.", "Textbook single-leg squat!", "Wow, perfect depth!"])
                
                stage = 'up'
                lowest_angle = 180
                worst_posture = 180
            elif stage == 'up' and status not in ["Amazing balance! Great pistol squat.", "Textbook single-leg squat!", "Wow, perfect depth!"]:
                if reps == 0: status = "Balance on one leg. Let's see that pistol squat!"
                
        elif active_knee < 100: 
            stage = 'down'
            status = "Hold... drive up through your heel!"
        elif 100 <= active_knee <= 160:
            if stage == 'up': status = "Control the descent, keep your balance..."

        # Cool down fatigue slowly if no alerts are active
        if not fatigue_alerts:
            fatigue_index = max(0.0, fatigue_index - 0.2)
            
        # Cap fatigue at 100
        fatigue_index = min(100.0, fatigue_index)
        
        # Update trackers
        state['prev_hip_y'] = hip_y
        state['prev_time'] = current_time
        state['baseline_vel'] = baseline_vel


    # Placeholder for other modules
    elif exercise_type in ["cardio", "yoga"]:
        status = "Module currently under construction. Please use Push-ups or Squats."

    # Update state dictionary
    state['stage'] = stage
    state['reps'] = reps
    state['lowest_angle'] = lowest_angle
    state['worst_posture'] = worst_posture
    state['status'] = status
    
    # Save Fatigue variables
    state['fatigue_index'] = fatigue_index
    state['fatigue_alerts'] = fatigue_alerts

    return state, play_audio