import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def detect_exercise(exercise_type, landmarks, current_stage, rep_count):
    """Routes the landmark data to the correct exercise logic."""
    play_audio = False
    status = "Tracking..."

    if exercise_type == "pushup":
        # Tracks Shoulder -> Elbow -> Wrist
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle < 90: 
            current_stage = "down"
        if angle > 160 and current_stage == 'down':
            current_stage = "up"
            rep_count += 1
            play_audio = True
            
        status = "Push Now!" if current_stage == "up" else "Down!"

    elif exercise_type == "squat":
        # Tracks Hip -> Knee -> Ankle
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        angle = calculate_angle(hip, knee, ankle)
        
        if angle < 100: # Deep squat
            current_stage = "down"
        if angle > 160 and current_stage == 'down':
            current_stage = "up"
            rep_count += 1
            play_audio = True
            
        status = "Squat!" if current_stage == "up" else "Stand Up!"

    elif exercise_type == "cardio":
        # Tracks Jumping Jacks (Wrists going above nose)
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        
        # Arms are up
        if left_wrist_y < nose_y and right_wrist_y < nose_y:
            current_stage = "up"
        # Arms are down
        if left_wrist_y > shoulder_y and right_wrist_y > shoulder_y and current_stage == 'up':
            current_stage = "down"
            rep_count += 1
            play_audio = True
            
        status = "Jump!" if current_stage == "down" else "Arms Down!"

    elif exercise_type == "yoga":
        # Tracks Sun Salutation (Deep reach up, then bend to knees/toes)
        wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        
        # Reaching high up
        if wrist_y < shoulder_y - 0.2: 
            current_stage = "up"
        # Bending down to knees
        if wrist_y > knee_y and current_stage == 'up':
            current_stage = "down"
            rep_count += 1
            play_audio = True
            
        status = "Reach Up!" if current_stage == "down" else "Bend Down!"

    return rep_count, current_stage, status, play_audio