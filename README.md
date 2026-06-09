# Exercise-form-check-and-rep-counter-using-Blazepose
Computer Vision pipeline built on top of MediaPipe BlazePose. This assists the user in checking their form is correct for a particular exercise and counts reps with proper form.

##  Key Architectural Features

* **Strict Configuration Decoupling:** Zero hardcoded mathematical fallback coordinates in the core runtime. Skeletons, tolerances, and calibration limits are managed entirely through an external JSON orchestration matrix.
* **Scale-Invariant Kinematics:** Joint angles are extracted using normalized 3D spatial vector dot products, making the tracker invariant to camera distance, camera angles, and user body dimensions.
* **Hysteresis-Driven Multi-Channel Tracking:** Dynamically evaluates bilateral joint visibility metrics to lock onto the optimal profile view, filtering out tracking jitter and rapid side-to-side camera flipping.
* **Dual-Mode Execution Engine:** Supports instant runtime execution using persisted historical baselines or structural snapshot-agnostic calibration.

##  Quick Start

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
