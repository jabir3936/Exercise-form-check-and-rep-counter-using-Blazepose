import cv2
import mediapipe as mp
import numpy as np
import time
import json
import argparse
import sys
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

model_path = 'pose_landmarker_full.task' 
DEFAULTS_FILE = 'exercise_configs.json'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_last_active_side = None

# PERSISTENCE DATA STORE STORAGE

def load_persisted_defaults():
    """Loads system profile thresholds directly from the external configuration file."""
    if not os.path.exists(DEFAULTS_FILE):
        print(f"ERROR: Master config file '{DEFAULTS_FILE}' is missing.")
        sys.exit(1)
    try:
        with open(DEFAULTS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed parsing '{DEFAULTS_FILE}': {e}")
        sys.exit(1)

def save_exercise_profile(exercise_name, updated_profile):
    """Overwrites the baseline profiles directly inside the configuration file."""
    master_store = load_persisted_defaults()
    ex_key = exercise_name.lower()
    
    # Update only the calibration components into the file's target exercise dictionary
    master_store[ex_key]["LEFT_SIDE"] = updated_profile["LEFT_SIDE"]
    master_store[ex_key]["RIGHT_SIDE"] = updated_profile["RIGHT_SIDE"]
    
    with open(DEFAULTS_FILE, 'w') as f:
        json.dump(master_store, f, indent=4)
    print(f"Completed, updated angle thresholds inside '{DEFAULTS_FILE}' for [{exercise_name.upper()}].")

# KINEMATIC MATHEMATICS ENGINE

def get_angle(landmarks, idx_a, idx_b, idx_c):
   #Calculates scale-invariant 3D joint angle at vertex B.
    a = np.array([landmarks[idx_a].x, landmarks[idx_a].y, landmarks[idx_a].z])
    b = np.array([landmarks[idx_b].x, landmarks[idx_b].y, landmarks[idx_b].z])
    c = np.array([landmarks[idx_c].x, landmarks[idx_c].y, landmarks[idx_c].z])
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

def get_torso_tilt(landmarks, idx_shoulder, idx_hip):
    #Calculates body orientation relative to gravity.
    s = np.array([landmarks[idx_shoulder].x, landmarks[idx_shoulder].y])
    h = np.array([landmarks[idx_hip].x, landmarks[idx_hip].y])
    return float(np.degrees(np.arctan2(abs(s[1] - h[1]), abs(s[0] - h[0]))))

def get_best_tracking_side(landmarks, left_group, right_group):
    #Evaluates the optimal side using tracking visibility and hysteresis.
    global _last_active_side
    left_mean_vis = np.mean([landmarks[i].visibility for i in left_group])
    right_mean_vis = np.mean([landmarks[i].visibility for i in right_group])
    
    switch_margin = 0.08 
    if _last_active_side == "LEFT_SIDE":
        if right_mean_vis > (left_mean_vis + switch_margin): _last_active_side = "RIGHT_SIDE"
    elif _last_active_side == "RIGHT_SIDE":
        if left_mean_vis > (right_mean_vis + switch_margin): _last_active_side = "LEFT_SIDE"
    else:
        _last_active_side = "LEFT_SIDE" if left_mean_vis >= right_mean_vis else "RIGHT_SIDE"
        
    return (left_group, "LEFT_SIDE", left_mean_vis) if _last_active_side == "LEFT_SIDE" else (right_group, "RIGHT_SIDE", right_mean_vis)

def extract_snapshot_agnostic(img_path, left_joints, right_joints, left_orient, right_orient):
    #Extracts reference angles from a calibration image.
    img = cv2.imread(img_path)
    if img is None: return None
        
    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.IMAGE)
    with PoseLandmarker.create_from_options(options) as detector:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_image)
        if not res.pose_world_landmarks: return None
        
        wlms = res.pose_world_landmarks[0]
        left_angle = get_angle(wlms, *left_joints)
        left_tilt = get_torso_tilt(wlms, *left_orient)
        right_angle = get_angle(wlms, *right_joints)
        right_tilt = get_torso_tilt(wlms, *right_orient)
        
        left_vis = np.mean([wlms[i].visibility for i in left_joints])
        right_vis = np.mean([wlms[i].visibility for i in right_joints])
        
        if left_vis >= right_vis: right_angle, right_tilt = left_angle, left_tilt
        else: left_angle, left_tilt = right_angle, right_tilt
        
        return {
            "LEFT_SIDE": {"angle": left_angle, "tilt": left_tilt},
            "RIGHT_SIDE": {"angle": right_angle, "tilt": right_tilt}
        }

def build_custom_profile(name, meta_config, start_img, mid_img, target_img):
    #Extracts raw imagery features to construct a temporary runtime configuration profile.
    l_j, r_j = meta_config["left_joints"], meta_config["right_joints"]
    l_o, r_o = meta_config["left_orient"], meta_config["right_orient"]
    
    print(f"Calibrating Custom Bilateral Profile via Snapshot Assets: [{name.upper()}]...")
    start_data = extract_snapshot_agnostic(start_img, l_j, r_j, l_o, r_o)
    mid_data = extract_snapshot_agnostic(mid_img, l_j, r_j, l_o, r_o)
    target_data = extract_snapshot_agnostic(target_img, l_j, r_j, l_o, r_o)
    
    if not all([start_data, mid_data, target_data]):
        print("CRITICAL: Snapshot feature calibration extraction failed completely. Exiting pipeline.")
        sys.exit(1)

    left_dir = -1 if start_data["LEFT_SIDE"]["angle"] > target_data["LEFT_SIDE"]["angle"] else 1
    right_dir = -1 if start_data["RIGHT_SIDE"]["angle"] > target_data["RIGHT_SIDE"]["angle"] else 1
    
    runtime_profile = meta_config.copy()
    runtime_profile.update({
        "name": name.upper(), 
        "LEFT_SIDE": {
            "start_angle": start_data["LEFT_SIDE"]["angle"],
            "mid_angle": mid_data["LEFT_SIDE"]["angle"],
            "target_angle": target_data["LEFT_SIDE"]["angle"],
            "torso_baseline": start_data["LEFT_SIDE"]["tilt"],
            "direction": left_dir
        },
        "RIGHT_SIDE": {
            "start_angle": start_data["RIGHT_SIDE"]["angle"],
            "mid_angle": mid_data["RIGHT_SIDE"]["angle"],
            "target_angle": target_data["RIGHT_SIDE"]["angle"],
            "torso_baseline": start_data["RIGHT_SIDE"]["tilt"],
            "direction": right_dir
        }
    })
    return runtime_profile


# TRACKING ENGINE RUNTIME EXECUTION

def run_universal_tracker(video_source, exercise_profile, output_json_path):
    ep = exercise_profile
    try: source = int(video_source)
    except ValueError: source = video_source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"CRITICAL: Unable to initialize video source: {video_source}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    is_webcam = isinstance(source, int)
    frame_budget_ms = 1000 / fps if fps > 0 else 33.3

    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)

    current_state = "START"
    target_hit = False
    rep_count = 0
    frame_count = 0
    feedback = "Get Ready"
    tracking_valid = True
    
    total_rep_frames = 0
    valid_form_frames = 0
    target_angle_snapshot = 0.0  
    
    reps_list = []
    session_start_time = time.time()
    VISIBILITY_THRESHOLD = 0.50
    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
    start_time_ms = int(time.time() * 1000)

    while cap.isOpened():
        loop_start_time = time.time()
        success, frame = cap.read()
        if not success: break
        
        timestamp_ms = int(time.time() * 1000) - start_time_ms if is_webcam else int((frame_count / fps) * 1000)
        frame_count += 1
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        if detection_result.pose_world_landmarks and detection_result.pose_landmarks:
            wlms = detection_result.pose_world_landmarks[0]
            
            active_joints, active_side, mean_visibility = get_best_tracking_side(wlms, ep["left_joints"], ep["right_joints"])
            
            if mean_visibility < VISIBILITY_THRESHOLD:
                tracking_valid = False
                feedback = "TRACKING LOST: LOW CONFIDENCE"
            else:
                tracking_valid = True
                active_orient = ep["left_orient"] if active_side == "LEFT_SIDE" else ep["right_orient"]
                active_legs = ep["left_legs"] if active_side == "LEFT_SIDE" else ep["right_legs"]
                side_profile = ep[active_side]
                
                live_angle = get_angle(wlms, *active_joints)
                live_tilt = get_torso_tilt(wlms, *active_orient)
                
                spine_valid = abs(live_tilt - side_profile["torso_baseline"]) <= ep["leeway"]
                legs_straight = True
                live_knee_angle = None
                
                if active_legs is not None:
                    live_knee_angle = get_angle(wlms, *active_legs)
                    legs_straight = live_knee_angle >= 160.0
                    
                form_valid = spine_valid and legs_straight
                
                lw = ep["leeway"]
                m_ang = side_profile["mid_angle"]
                t_ang = side_profile["target_angle"]
                s_ang = side_profile["start_angle"]
                
                if side_profile["direction"] == -1:
                    moved_past_mid = live_angle < m_ang
                    reached_target  = live_angle <= (t_ang + lw)
                    returned_home   = live_angle >= (s_ang - lw)
                else:
                    moved_past_mid = live_angle > m_ang
                    reached_target  = live_angle >= (t_ang - lw)
                    returned_home   = live_angle <= (s_ang + lw)
                
                if current_state == "START":
                    if form_valid: feedback = f"Form OK ({active_side})"
                    else: feedback = "KEEP LEGS STRAIGHT!" if not legs_straight else "CORRECT POSTURE!"
                        
                    if moved_past_mid:
                        current_state = "MID_ACTION"
                        target_hit = False
                        total_rep_frames = 0
                        valid_form_frames = 0
                        target_angle_snapshot = live_angle
                        
                elif current_state == "MID_ACTION":
                    total_rep_frames += 1
                    if form_valid: valid_form_frames += 1
                    
                    if not form_valid: feedback = "KEEP LEGS STRAIGHT!" if not legs_straight else "CORRECT POSTURE!"
                    else: feedback = "Finish the rep!" if target_hit else "In motion..."
                    
                    if reached_target:
                        if not target_hit:
                            target_hit = True
                            target_angle_snapshot = live_angle
                        else:
                            if side_profile["direction"] == -1:
                                target_angle_snapshot = min(target_angle_snapshot, live_angle)
                            else:
                                target_angle_snapshot = max(target_angle_snapshot, live_angle)
                        
                    elif returned_home:
                        current_state = "START"
                        form_score = (valid_form_frames / total_rep_frames) if total_rep_frames > 0 else 1.0
                        
                        if target_hit and form_score >= 0.70:
                            rep_count += 1
                            feedback = "Good Rep!"
                            
                            reps_list.append({
                                "rep_number": rep_count,
                                "exercise": ep["name"].capitalize(),
                                "active_side": active_side,
                                "joint_angle_at_target": round(target_angle_snapshot, 1),
                                "form_score": round(form_score, 3),
                                "timestamp_ms": timestamp_ms
                            })
                        else:
                            feedback = "Rep Disallowed: Partial Depth" if not target_hit else "Rep Disallowed: Bad Form"
            
            for pose_landmarks in detection_result.pose_landmarks:
                drawing_utils.draw_landmarks(
                    image=frame, landmark_list=pose_landmarks,
                    connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=pose_landmark_style, connection_drawing_spec=pose_connection_style)
            
            hud_height = 155 if ('live_knee_angle' in locals() and live_knee_angle is not None) else 135
            hud_overlay = frame.copy()
            cv2.rectangle(hud_overlay, (10, 10), (460, hud_height), (15, 15, 15), -1)
            cv2.addWeighted(hud_overlay, 0.65, frame, 0.35, 0, frame)
            
            y_offset = 30
            cv2.putText(frame, f"EXERCISE: {ep['name']} ({current_state})", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            if tracking_valid:
                cv2.putText(frame, f"ACTIVE CHANNEL: {active_side} (Vis: {mean_visibility:.2f})", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                if live_knee_angle is not None:
                    y_offset += 20
                    cv2.putText(frame, f"KNEE ANGLE: {int(live_knee_angle)} DEG", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                status_color = (0, 255, 0) if form_valid else (0, 0, 255)
            else:
                cv2.putText(frame, f"ACTIVE CHANNEL: TRACKING CORRUPTED", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                status_color = (0, 0, 255)
                
            y_offset += 30
            cv2.putText(frame, f"REPS: {rep_count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            y_offset += 35
            cv2.putText(frame, f"STATUS: {feedback}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
        cv2.imshow("Universal Kinematics Engine", frame)
        elapsed_ms = (time.time() - loop_start_time) * 1000
        delay = max(1, int(frame_budget_ms - elapsed_ms)) if not is_webcam else 1
        if cv2.waitKey(delay) & 0xFF == ord('q'): break
        
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    total_duration_seconds = round(time.time() - session_start_time, 1)

    session_payload = {
        "exercise": ep["name"].capitalize(),
        "total_reps": rep_count,
        "session_duration_seconds": total_duration_seconds,
        "reps": reps_list
    }
        
    with open(output_json_path, 'w') as f:
        json.dump(session_payload, f, indent=2)
    print(f"\n Session summary metrics written to: [{output_json_path}]")
    
    return rep_count

# SYSTEM CONTEXT INTERFACE DISPATCHER
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Kinematics Engine")
    parser.add_argument("--mode", type=str, default="default", choices=["default", "calibrate"])
    parser.add_argument("--exercise", type=str, required=True, choices=["pushup", "pullup", "chinup"])
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--output", type=str, default="session_output.json")
    
    parser.add_argument("--start_img", type=str, default=None)
    parser.add_argument("--mid_img", type=str, default=None)
    parser.add_argument("--target_img", type=str, default=None)
    
    args = parser.parse_args()
    
    # Read settings entirely from external file store layout matrix
    db_configs = load_persisted_defaults()
    selected_config = db_configs[args.exercise.lower()]

    if args.mode == "calibrate":
        if not all([args.start_img, args.mid_img, args.target_img]):
            print("ERROR: '--mode calibrate' requires snapshots: --start_img, --mid_img, and --target_img.")
            sys.exit(1)
        active_profile = build_custom_profile(args.exercise, selected_config, args.start_img, args.mid_img, args.target_img)
    else:
        print(f"Loading configurations for [{args.exercise.upper()}] directly from '{DEFAULTS_FILE}'...")
        active_profile = selected_config
        active_profile["name"] = args.exercise.upper()

    # Run execution pipeline
    reps_completed = run_universal_tracker(video_source=args.source, exercise_profile=active_profile, output_json_path=args.output)
    
    if args.mode == "calibrate" and reps_completed > 0:
        print("\n" + "="*60)
        user_choice = input(f"Are you satisfied with how the {reps_completed} reps were evaluated? Save as future system default? (y/n): ").strip().lower()
        if user_choice in ['y', 'yes']:
            save_exercise_profile(args.exercise, active_profile)
        else:
            print("Configuration discarded. Stored configuration maps remain untouched.")