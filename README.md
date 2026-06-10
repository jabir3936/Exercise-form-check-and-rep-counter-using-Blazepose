# Exercise-Form Check and Rep Counter Using Blazepose

A production-ready, Computer Vision pipeline built on top of MediaPipe BlazePose. This system assists users by evaluating human telemetry in real time, verifying execution form against anatomical benchmarks, and tracking precise repetition counts using an isolated state machine.

<img width="800" height="544" alt="ezgif-6d4740da39616f3f" src="https://github.com/user-attachments/assets/66de88f2-bb9f-42cd-a6cf-ba9d894af643" />

---

##  Key Architectural Features

* **Strict Configuration Decoupling:** Zero hardcoded mathematical fallback coordinates in the core runtime. Skeletons, tolerances, and calibration limits are managed entirely through an external JSON.
* **Scale-Invariant Kinematics:** Joint angles are extracted using normalized 3D spatial vector dot products, making the tracker invariant to camera distance, camera angles, and user body dimensions.
* **Hysteresis-Driven Multi-Channel Tracking:** Dynamically evaluates bilateral joint visibility metrics to lock onto the optimal profile view, filtering out tracking jitter and rapid side-to-side camera flipping.
* **Dual-Mode Execution Engine:** Supports instant runtime execution using persisted historical baselines or structural snapshot-agnostic calibration.

---

##  Quick Start

### 1. Environment Setup
Clone the repository and install the verified dependency versions:
```bash
git clone [https://github.com/YOUR_USERNAME/Exercise-form-check-and-rep-counter-using-Blazepose.git](https://github.com/YOUR_USERNAME/Exercise-form-check-and-rep-counter-using-Blazepose.git)
cd Exercise-form-check-and-rep-counter-using-Blazepose
pip install -r requirements.txt
```

### 2. Download the Model Weights
The pipeline requires MediaPipe's asset bundle to calculate coordinates. Download the model and place it directly into your root project directory:
* [Download MediaPipe Pose Landmarker (Full Model)](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)
* Ensure the downloaded file is saved exactly as: `pose_landmarker_full.task`

### 3. Execution
Run the live tracker against an existing video file or stream directly from your local webcam:

#### Run via Webcam (Source 0)
```bash
python exercise_pose.py --mode default --exercise pullup --source 0
```

#### Run via Video File Path
```bash
python exercise_pose.py --mode default --exercise pullup --source "your_pullup_video.mp4"
```

Currently calibrated exercises out-of-the-box inside `exercise_configs.json`: **pullups** and **pushups**.

---

##  Calibration & Custom Exercise Creation

To add a new movement pattern to the orchestration layer, you must first register your target joint topologies inside `exercise_configs.json`.

### 1. Define Keypoint Topologies
Map your structural coordinates using the standard **MediaPipe BlazePose Landmark Schema**:

<img width="1999" height="1010" alt="image4 (1)" src="https://github.com/user-attachments/assets/dff26ffc-94d0-4b07-8c30-d2454a7154e3" />


Add your custom profile block to the JSON file. You can pass placeholder values (`0.0`) for the inner execution limits (`start_angle`, `mid_angle`, `target_angle`, `torso_baseline`), as the engine will calculate these automatically during the calibration phase:

```json
"squat": {
    "left_joints": [23, 25, 27],
    "right_joints": [24, 26, 28],
    "left_orient": [11, 23],
    "right_orient": [12, 24],
    "left_legs": [23, 25, 27],
    "right_legs": [24, 26, 28],
    "leeway": 15.0,
    "LEFT_SIDE": {
        "start_angle": 0.0, "mid_angle": 0.0, "target_angle": 0.0, "torso_baseline": 0.0, "direction": 1
    },
    "RIGHT_SIDE": {
        "start_angle": 0.0, "mid_angle": 0.0, "target_angle": 0.0, "torso_baseline": 0.0, "direction": 1
    }
}
```

* **`joints`**: The vertex triad tracked to register a rep (e.g., Hip -> Knee -> Ankle for squats).
* **`orient`**: The vertical tracking coordinates (Shoulder -> Hip) used to lock down trunk posture baselines.
* **`legs`**: Optional checking array used to enforce rigid extensions (set to `null` if unnecessary for the movement type).
* **`leeway`**: Structural angular deviation margin allowed before triggering form errors.

### 2. Execute Calibration Routine
Run the engine in `--mode calibrate` by passing three snapshot assets capturing the distinct biometric execution phases of the rep:

```bash
python exercise_pose.py --mode calibrate --exercise squat --start_img squat_start.jpg --mid_img squat_eccentric.jpg --target_img squat_target.jpg --source squat_test_run.mp4
```

1. **`--start_img`**: The initial starting frame setup (e.g., standing tall).
2. **`--mid_img`**: The absolute midway transition point of the movement.
3. **`--target_img`**: The peak range-of-motion target depth frame (e.g., maximum depth of a squat).

At the conclusion of the video processing stream, typing `y` at the terminal prompt will automatically write and persist these exact calculated kinematics to your configuration matrix for permanent use.

---

## 📊 Standardized Telemetry Export

Upon exiting the application window, session metadata is compiled into a structured JSON string output saved to your designated `--output` path. This is optimized for direct consumption by web dashboards or database backends:

```json
{
  "exercise": "Pushup",
  "total_reps": 4,
  "session_duration_seconds": 15.4,
  "reps": [
    {
      "rep_number": 1,
      "exercise": "Pushup",
      "active_side": "RIGHT_SIDE",
      "joint_angle_at_target": 86.4,
      "form_score": 1.0,
      "timestamp_ms": 2877
    },
    {
      "rep_number": 2,
      "exercise": "Pushup",
      "active_side": "RIGHT_SIDE",
      "joint_angle_at_target": 82.6,
      "form_score": 1.0,
      "timestamp_ms": 6589
    },
    {
      "rep_number": 3,
      "exercise": "Pushup",
      "active_side": "RIGHT_SIDE",
      "joint_angle_at_target": 82.7,
      "form_score": 0.94,
      "timestamp_ms": 9884
    },
    {
      "rep_number": 4,
      "exercise": "Pushup",
      "active_side": "RIGHT_SIDE",
      "joint_angle_at_target": 86.2,
      "form_score": 1.0,
      "timestamp_ms": 12721
    }
  ]
}
```

---

##  License

This software framework is distributed under the **MIT License**. It grants full commercial execution permissions, modifications, and closed-source private integration structures without legal restrictions, while providing comprehensive developer liability protection.
