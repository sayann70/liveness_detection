import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from collections import deque
from app.database import SessionLocal
from app.services.db_services import (
    start_new_state_session, 
    close_active_session, 
    resolve_orphaned_sessions, 
    get_or_create_default_employee
)
from app.config import settings

def draw_rounded_rect(img, pt1, pt2, color, thickness, r):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    w = x2 - x1
    h = y2 - y1
    r = int(min(r, w // 2, h // 2))
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)

def draw_filled_rounded_rect(img, pt1, pt2, color, r):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    w = x2 - x1
    h = y2 - y1
    r = int(min(r, w // 2, h // 2))
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1, lineType=cv2.LINE_AA)

class CVMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.latest_frame = None  # Holds JPEG bytes
        
        # State machine variables (thread-safe protected by lock)
        self.current_status = "ABSENT"
        self.confidence = 0.0
        self.last_present_time = time.time()
        self.last_movement_time = time.time()
        self.state_entered_time = time.time()
        
        # Diagnostics score tracking
        self.latest_raw_score = 0.0
        self.latest_smoothed_score = 0.0
        self.frame_scores = deque(maxlen=150)  # ~10 second rolling window at 15 FPS
        
        self.lock = threading.Lock()
        self.is_mock = False
        self.prev_landmarks = {}
        
        # Active employee dynamic tracking
        self.active_employee_id = settings.DEFAULT_EMPLOYEE_ID
        
        # Eye blink tracking variables
        self.blink_count = 0
        self.prev_blinking = False
        
        # YOLO tracking and smoothing refinements
        self.smoothed_landmarks = {}
        self.smoothed_kp_xy = None
        self.locked_track_id = None
        self.smoothed_bbox = None
        self.active_employee_name = "Ankur Bag"
        self._update_employee_name_cache()

    def _update_employee_name_cache(self):
        db = SessionLocal()
        try:
            from app.models import Employee
            emp = db.query(Employee).filter(Employee.employee_id == self.active_employee_id).first()
            if emp:
                self.active_employee_name = emp.name
            else:
                self.active_employee_name = f"Subject #{self.active_employee_id}"
        except Exception as e:
            print(f"[CV Engine] Cache Name Error: {e}")
            self.active_employee_name = f"Subject #{self.active_employee_id}"
        finally:
            db.close()



    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def get_current_state(self):
        with self.lock:
            return {
                "status": self.current_status,
                "confidence": self.confidence,
                "time_in_state": int(time.time() - self.state_entered_time),
                "is_mock": self.is_mock
            }

    def get_debug_metrics(self) -> dict:
        """Returns live values for the developer diagnostic dashboard."""
        with self.lock:
            now = time.time()
            idle_cd = 0
            
            if self.current_status == "WORKING":
                elapsed_low = now - self.last_movement_time
                # Countdown from 60 seconds of continuous inactivity
                idle_cd = max(0, int(settings.IDLE_TIMEOUT - elapsed_low))
                
            return {
                "raw_score": round(self.latest_raw_score, 3),
                "smoothed_score": round(self.latest_smoothed_score, 3),
                "movement_threshold": settings.YOLO_ACTIVITY_THRESHOLD,  # Boundary between Low and Working
                "idle_countdown": idle_cd,
                "working_countdown": 0,
                "epsilon_filter": settings.YOLO_EPSILON_FILTER
            }

    def set_active_employee(self, employee_id: int):
        """Thread-safe method to change the actively monitored employee and transition session segments."""
        with self.lock:
            if self.active_employee_id == employee_id:
                return
            
            print(f"[CV Engine] Switching active monitored subject from ID {self.active_employee_id} to ID {employee_id}")
            
            db = SessionLocal()
            try:
                # 1. Close current active session for the old employee
                close_active_session(db, self.active_employee_id)
                
                # 2. Update active employee ID
                self.active_employee_id = employee_id
                self._update_employee_name_cache()
                
                # 3. Start a new session for the new employee with current state
                start_new_state_session(
                    db,
                    self.active_employee_id,
                    self.current_status,
                    self.confidence,
                    raw_score=self.latest_raw_score,
                    smoothed_score=self.latest_smoothed_score,
                    transition_reason=f"Switched monitoring subject to employee ID {employee_id}.",
                    notes=f"Switched monitoring subject to employee ID {employee_id}."
                )
            except Exception as e:
                print(f"[CV Engine] Error switching active employee: {e}")
            finally:
                db.close()
                
            # Clear frame scores for a clean slate
            self.frame_scores.clear()
            self.prev_landmarks = {}
            self.blink_count = 0
            self.prev_blinking = False

    def _run_loop(self):
        # 1. Resolve orphaned sessions and seed employee on startup
        db = SessionLocal()
        try:
            resolve_orphaned_sessions(db)
            employee = get_or_create_default_employee(db)
            # Use the dynamically selected active employee ID
            employee_id = self.active_employee_id
            self._update_employee_name_cache()
            
            # Start initial ABSENT session segment in DB
            start_new_state_session(
                db, 
                employee_id, 
                self.current_status, 
                self.confidence, 
                raw_score=0.0,
                smoothed_score=0.0,
                transition_reason="Initial state on monitoring start.",
                notes="Initial state on monitoring start."
            )
        except Exception as e:
            print(f"[CV Engine] Error during startup database prep: {e}")
        finally:
            db.close()

        # 2. Attempt webcam connection
        cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        
        # Set resolution to 640x480 for fast processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened() or settings.MOCK_CAMERA:
            print(f"[CV Engine] Webcam index {settings.CAMERA_INDEX} not available or mock mode forced. Starting simulated feed...")
            self.is_mock = True
        else:
            print(f"[CV Engine] Successfully opened webcam on index {settings.CAMERA_INDEX}")
            self.is_mock = False

        # 3. Setup YOLOv8 Pose model from configuration
        try:
            print(f"[CV Engine] Initializing YOLOv8 Pose model ({settings.YOLO_MODEL_NAME}) on device '{settings.YOLO_DEVICE}'...")
            yolo_model = YOLO(settings.YOLO_MODEL_NAME)
            print(f"[CV Engine] YOLOv8 Pose model {settings.YOLO_MODEL_NAME} loaded successfully.")
        except Exception as e:
            print(f"[CV Engine] Failed to load YOLOv8 model: {e}. Falling back to Mock Mode.")
            self.is_mock = True
            yolo_model = None

        prev_time = time.time()

        # Main processing loop
        try:
            while self.running:
                loop_start = time.time()
                frame = None
                detected = False
                model_confidence = 0.0
                current_landmarks = {}
                is_blinking = False
                ear = 0.3

                if self.is_mock:
                    # Mock Mode - Generate synthetic frame
                    frame, detected, model_confidence, current_landmarks = self._generate_mock_frame()
                    
                    # Simulate periodic blinks in mock mode
                    cycle_time = int(time.time()) % 90
                    if cycle_time < 70:  # User present
                         sec_of_cycle = time.time() % 4
                         is_blinking = (sec_of_cycle < 0.25)  # Blink for 0.25s every 4s
                    
                    time.sleep(max(0.01, (1.0 / 15.0) - (time.time() - loop_start)))  # Limit mock FPS to ~15
                else:
                    # WebCam Mode - Read hardware frame
                    ret, frame = cap.read()
                    if not ret:
                        print("[CV Engine] Failed to grab frame. Falling back to Mock Mode...")
                        self.is_mock = True
                        continue

                    # Mirror frame for intuitive viewing
                    frame = cv2.flip(frame, 1)
                    h, w, c = frame.shape

                    # Convert to RGB for YOLOv8
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run detections using YOLOv8
                    if yolo_model is None:
                        self.is_mock = True
                        continue
                    
                    # Run tracking with configured parameters for real-time consistency
                    yolo_results = yolo_model.track(
                        rgb_frame,
                        persist=True,
                        tracker=settings.YOLO_TRACKER,
                        imgsz=settings.YOLO_INFERENCE_SIZE,
                        conf=settings.YOLO_CONFIDENCE_THRESHOLD,
                        device=settings.YOLO_DEVICE,
                        verbose=False
                    )
                    
                    pose_detected = False
                    face_detected = False
                    detected = False
                    model_confidence = 0.0
                    is_blinking = False  # Blink is simulated in mock mode, disabled in webcam mode

                    if yolo_results and len(yolo_results[0].boxes) > 0:
                        boxes = yolo_results[0].boxes
                        
                        # Primary user tracking index selection
                        # Calculate box areas (xyxy coordinates)
                        xyxy_array = boxes.xyxy.cpu().numpy()
                        areas = (xyxy_array[:, 2] - xyxy_array[:, 0]) * (xyxy_array[:, 3] - xyxy_array[:, 1])
                        
                        # Get tracker IDs if available
                        track_ids = None
                        if boxes.id is not None:
                            track_ids = boxes.id.cpu().numpy().astype(int)
                        
                        # Select best detection target matching locked track ID, or largest box
                        primary_idx = 0
                        if track_ids is not None and self.locked_track_id is not None and self.locked_track_id in track_ids:
                            primary_idx = int(np.where(track_ids == self.locked_track_id)[0][0])
                        else:
                            primary_idx = int(np.argmax(areas))
                            if track_ids is not None:
                                self.locked_track_id = int(track_ids[primary_idx])
                        
                        pose_detected = True
                        face_detected = True
                        detected = True
                        
                        kp_xyn = yolo_results[0].keypoints.xyn[primary_idx].cpu().numpy()  # shape (17, 2)
                        kp_xy = yolo_results[0].keypoints.xy[primary_idx].cpu().numpy()    # shape (17, 2)
                        kp_conf = yolo_results[0].keypoints.conf[primary_idx].cpu().numpy()  # shape (17,)
                        
                        model_confidence = float(np.mean(kp_conf))

                        # Exponential Moving Average keypoint smoothing to eliminate coordinate jitter
                        alpha = settings.YOLO_SMOOTHING_FACTOR
                        if self.smoothed_kp_xy is None or len(self.smoothed_kp_xy) != len(kp_xy):
                            self.smoothed_kp_xy = kp_xy.copy()
                        else:
                            for idx in range(len(kp_xy)):
                                if kp_conf[idx] > 0.3:
                                    self.smoothed_kp_xy[idx] = alpha * kp_xy[idx] + (1 - alpha) * self.smoothed_kp_xy[idx]
                                else:
                                    self.smoothed_kp_xy[idx] = kp_xy[idx]

                        # Gather landmarks for movement analysis using normalized coordinates
                        # YOLO keypoints index: 0=nose, 5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow, 9=L_wrist, 10=R_wrist
                        # MediaPipe-like mapping: nose=0, shoulders=11/12, elbows=13/14, wrists=15/16
                        for yolo_idx, mp_idx in [(0, 0), (5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]:
                            if kp_conf[yolo_idx] > 0.3:
                                current_landmarks[mp_idx] = (float(kp_xyn[yolo_idx][0]), float(kp_xyn[yolo_idx][1]))
                        
                        # Smooth landmarks for movement analysis
                        for mp_idx, val in current_landmarks.items():
                            if mp_idx in self.smoothed_landmarks:
                                prev_val = self.smoothed_landmarks[mp_idx]
                                self.smoothed_landmarks[mp_idx] = (
                                    alpha * val[0] + (1 - alpha) * prev_val[0],
                                    alpha * val[1] + (1 - alpha) * prev_val[1]
                                )
                            else:
                                self.smoothed_landmarks[mp_idx] = val
                        
                        # Draw thin, clean pose skeleton: Off-white lines, clean green joints using smoothed coordinates
                        connections = [
                            (5, 7), (7, 9),   # Left arm
                            (6, 8), (8, 10),  # Right arm
                            (11, 13), (13, 15), # Left leg
                            (12, 14), (14, 16)  # Right leg
                        ]
                        
                        for pt1_idx, pt2_idx in connections:
                            if kp_conf[pt1_idx] > 0.3 and kp_conf[pt2_idx] > 0.3:
                                p1 = (int(self.smoothed_kp_xy[pt1_idx][0]), int(self.smoothed_kp_xy[pt1_idx][1]))
                                p2 = (int(self.smoothed_kp_xy[pt2_idx][0]), int(self.smoothed_kp_xy[pt2_idx][1]))
                                cv2.line(frame, p1, p2, (235, 235, 235), 1)
                                
                        # Torso cage cross-bracing and midline structural lines
                        if kp_conf[5] > 0.3 and kp_conf[6] > 0.3 and kp_conf[11] > 0.3 and kp_conf[12] > 0.3:
                            s_mid_x = int((self.smoothed_kp_xy[5][0] + self.smoothed_kp_xy[6][0]) / 2.0)
                            s_mid_y = int((self.smoothed_kp_xy[5][1] + self.smoothed_kp_xy[6][1]) / 2.0)
                            h_mid_x = int((self.smoothed_kp_xy[11][0] + self.smoothed_kp_xy[12][0]) / 2.0)
                            h_mid_y = int((self.smoothed_kp_xy[11][1] + self.smoothed_kp_xy[12][1]) / 2.0)
                            
                            # Spine line
                            cv2.line(frame, (s_mid_x, s_mid_y), (h_mid_x, h_mid_y), (235, 235, 235), 1)
                            
                            # Chest cross braces
                            p_s5 = (int(self.smoothed_kp_xy[5][0]), int(self.smoothed_kp_xy[5][1]))
                            p_s6 = (int(self.smoothed_kp_xy[6][0]), int(self.smoothed_kp_xy[6][1]))
                            p_h11 = (int(self.smoothed_kp_xy[11][0]), int(self.smoothed_kp_xy[11][1]))
                            p_h12 = (int(self.smoothed_kp_xy[12][0]), int(self.smoothed_kp_xy[12][1]))
                            cv2.line(frame, p_s5, p_h12, (220, 220, 220), 1)
                            cv2.line(frame, p_s6, p_h11, (220, 220, 220), 1)
                            
                            # Outer sides
                            cv2.line(frame, p_s5, p_h11, (235, 235, 235), 1)
                            cv2.line(frame, p_s6, p_h12, (235, 235, 235), 1)
                            # Top shoulders and bottom hips
                            cv2.line(frame, p_s5, p_s6, (235, 235, 235), 1)
                            cv2.line(frame, p_h11, p_h12, (235, 235, 235), 1)
                            
                            # Mid-torso horizontal line
                            mt_l_x = int((self.smoothed_kp_xy[5][0] + self.smoothed_kp_xy[11][0]) / 2.0)
                            mt_l_y = int((self.smoothed_kp_xy[5][1] + self.smoothed_kp_xy[11][1]) / 2.0)
                            mt_r_x = int((self.smoothed_kp_xy[6][0] + self.smoothed_kp_xy[12][0]) / 2.0)
                            mt_r_y = int((self.smoothed_kp_xy[6][1] + self.smoothed_kp_xy[12][1]) / 2.0)
                            cv2.line(frame, (mt_l_x, mt_l_y), (mt_r_x, mt_r_y), (235, 235, 235), 1)
                            
                            # Collar lines (Nose to shoulders midpoint)
                            if kp_conf[0] > 0.3:
                                p_nose = (int(self.smoothed_kp_xy[0][0]), int(self.smoothed_kp_xy[0][1]))
                                cv2.line(frame, p_nose, (s_mid_x, s_mid_y), (235, 235, 235), 1)

                        # Draw joints as micro green circles
                        for idx in range(17):
                            if kp_conf[idx] > 0.3:
                                pt = (int(self.smoothed_kp_xy[idx][0]), int(self.smoothed_kp_xy[idx][1]))
                                cv2.circle(frame, pt, 1, (74, 163, 22), -1)

                        # Draw advanced palm-arch hand and finger skeletons
                        self._draw_hand_skeleton(frame, self.smoothed_kp_xy[9], self.smoothed_kp_xy[7], kp_conf[9], kp_conf[7])  # Left hand
                        self._draw_hand_skeleton(frame, self.smoothed_kp_xy[10], self.smoothed_kp_xy[8], kp_conf[10], kp_conf[8]) # Right hand

                        # Draw sleek, thin face skeleton mesh
                        self._draw_face_mesh(
                            frame,
                            self.smoothed_kp_xy[0],  # Nose
                            self.smoothed_kp_xy[3],  # Left Ear
                            self.smoothed_kp_xy[4],  # Right Ear
                            self.smoothed_kp_xy[1],  # Left Eye
                            self.smoothed_kp_xy[2],  # Right Eye
                            kp_conf[0], kp_conf[3], kp_conf[4], kp_conf[1], kp_conf[2]
                        )

                # Process blink transitions & reset countdown
                if is_blinking:
                    with self.lock:
                        if not self.prev_blinking:
                            self.blink_count += 1
                            self.last_movement_time = time.time()  # Keep user active!
                            print(f"[CV Engine] Blink detected (Count: {self.blink_count})")
                        self.prev_blinking = True
                else:
                    with self.lock:
                        self.prev_blinking = False

                # 4. Weighted Landmark Activity Scoring with Epsilon Filter
                raw_score = 0.0
                epsilon = settings.YOLO_EPSILON_FILTER
                
                # Group configs with target weights
                groups = {
                    "hands": {"indices": [15, 16], "weight": 0.50},      # Wrists (50%)
                    "arms": {"indices": [13, 14], "weight": 0.25},       # Elbows (25%)
                    "shoulders": {"indices": [11, 12], "weight": 0.15},  # Shoulders (15%)
                    "head": {"indices": [0], "weight": 0.10}            # Nose/Face (10%)
                }
                
                group_displacements = {}
                if detected and self.smoothed_landmarks and self.prev_landmarks:
                    for group_name, info in groups.items():
                        sum_d = 0.0
                        count = 0
                        for idx in info["indices"]:
                            if idx in self.smoothed_landmarks and idx in self.prev_landmarks:
                                pt = self.smoothed_landmarks[idx]
                                prev_pt = self.prev_landmarks[idx]
                                d = np.sqrt((pt[0] - prev_pt[0])**2 + (pt[1] - prev_pt[1])**2)
                                
                                # Apply Epsilon noise threshold filter
                                if d >= epsilon:
                                    sum_d += d
                                count += 1
                        if count > 0:
                            group_displacements[group_name] = sum_d / count

                # Normalize weights dynamically for visible groups
                if group_displacements:
                    total_available_weight = sum(groups[g]["weight"] for g in group_displacements.keys())
                    if total_available_weight > 0:
                        weighted_sum = 0.0
                        for g, disp in group_displacements.items():
                            normalized_w = groups[g]["weight"] / total_available_weight
                            weighted_sum += normalized_w * disp
                        # Scale raw score by 100 for readable 0.0 to 1.0+ range
                        raw_score = weighted_sum * 100.0

                # Give a boost to raw score if blink detected to help keep WORK state
                if detected and is_blinking:
                    raw_score += 0.8
                
                # Store current landmarks for next frame comparison
                if detected:
                    self.prev_landmarks = self.smoothed_landmarks.copy()
                else:
                    self.prev_landmarks = {}

                # 5. Temporal Rolling average Smoothing
                self.frame_scores.append(raw_score)
                smoothed_score = sum(self.frame_scores) / len(self.frame_scores) if self.frame_scores else 0.0
                
                with self.lock:
                    self.latest_raw_score = raw_score
                    self.latest_smoothed_score = smoothed_score

                # 6. State Machine Timer & Transition Logic
                now = time.time()
                new_status = self.current_status
                old_status = self.current_status
                state_changed = False
                transition_reason = ""

                with self.lock:
                    if detected:
                        self.last_present_time = now
                        self.confidence = float(model_confidence)
                        is_active_working = smoothed_score >= settings.YOLO_ACTIVITY_THRESHOLD  # Working threshold boundary

                        if self.current_status == "ABSENT":
                            # Return from absence -> Immediately WORKING
                            new_status = "WORKING"
                            self.last_movement_time = now
                            transition_reason = "Subject detected returned in frame."
                        elif is_active_working:
                            # Active movement resets the idle countdown
                            self.last_movement_time = now
                            if self.current_status == "IDLE":
                                new_status = "WORKING"
                                transition_reason = f"Active movement resumed (score: {smoothed_score:.3f})."
                        else:
                            # Low/No activity (score < YOLO_ACTIVITY_THRESHOLD)
                            if self.current_status == "WORKING":
                                # Verify if low activity persists for 60 seconds
                                elapsed_low = now - self.last_movement_time
                                if elapsed_low >= settings.IDLE_TIMEOUT:
                                    new_status = "IDLE"
                                    transition_reason = f"Low activity persisted for 60 seconds (average score: {smoothed_score:.3f})."
                    else:
                        # Person absent from view
                        self.confidence = 0.0
                        if self.current_status != "ABSENT":
                            # Verify if absence persists for 10 seconds
                            if now - self.last_present_time >= settings.ABSENT_TIMEOUT:
                                new_status = "ABSENT"
                                self.locked_track_id = None
                                self.smoothed_kp_xy = None
                                self.smoothed_landmarks = {}
                                transition_reason = "No person detected for 10 continuous seconds."

                    # Apply transition
                    if new_status != self.current_status:
                        print(f"[CV Engine] Transition: {self.current_status} -> {new_status} | Reason: {transition_reason}")
                        self.current_status = new_status
                        self.state_entered_time = now
                        state_changed = True

                # 7. Database log on state transition
                if state_changed:
                    db = SessionLocal()
                    try:
                        start_new_state_session(
                            db, 
                            self.active_employee_id, 
                            self.current_status, 
                            self.confidence,
                            raw_score=self.latest_raw_score,
                            smoothed_score=self.latest_smoothed_score,
                            transition_reason=transition_reason,
                            notes=transition_reason
                        )
                    except Exception as e:
                        print(f"[CV Engine] DB Session Log Error: {e}")
                    finally:
                        db.close()

                # 8. FPS calculation and HUD annotations
                current_fps = 1.0 / (time.time() - prev_time + 1e-6)
                prev_time = time.time()

                # Draw HUD panel on the frame
                self._draw_hud(frame, current_fps, raw_score, smoothed_score)

                # 9. Encode frame to JPEG bytes
                ret_enc, jpeg_buffer = cv2.imencode('.jpg', frame)
                if ret_enc:
                    self.latest_frame = jpeg_buffer.tobytes()

        finally:
            if not self.is_mock:
                cap.release()
            
            db = SessionLocal()
            try:
                close_active_session(db, self.active_employee_id)
                print("[CV Engine] Gracefully closed active session segment on shutdown.")
            except Exception as e:
                print(f"[CV Engine] DB shutdown segment close failed: {e}")
            finally:
                db.close()
            print("[CV Engine] Thread stopped successfully.")

    def _draw_hud(self, frame, fps, raw_score, smoothed_score):
        h, w, c = frame.shape
        # Solid background box for dashboard details
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 140), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Status text colors
        status_colors = {
            "WORKING": (74, 163, 22),  # #16A34A (Green)
            "IDLE": (11, 158, 245),    # #F59E0B (Amber)
            "ABSENT": (68, 68, 239)    # #EF4444 (Red)
        }
        color = status_colors.get(self.current_status, (255, 255, 255))

        # Add HUD Text (Professional and compact)
        cv2.putText(frame, f"STATE: {self.current_status}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if not self.is_mock and self.locked_track_id is not None:
            cv2.putText(frame, f"Track ID: {self.locked_track_id}", (180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Raw Score: {raw_score:.3f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Smooth Score: {smoothed_score:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.textWidth = 10
        
        # Idle/absent count downs
        now = time.time()
        if self.current_status == "WORKING":
            elapsed = now - self.last_movement_time
            cd = max(0, int(settings.IDLE_TIMEOUT - elapsed))
            cv2.putText(frame, f"Idle Countdown: {cd}s", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        else:
            cv2.putText(frame, f"Idle Countdown: --", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        mode_str = "SIMULATED FEED" if self.is_mock else "WEBCAM LIVE"
        cv2.putText(frame, f"Mode: {mode_str} ({fps:.1f} FPS)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Draw status circle indicator
        cv2.circle(frame, (w - 25, 25), 8, color, -1)
        cv2.circle(frame, (w - 25, 25), 10, (255, 255, 255), 1)

    def _draw_hand_skeleton(self, frame, w_pt, e_pt, w_conf, e_conf):
        if w_conf > 0.3 and e_conf > 0.3:
            dx = w_pt[0] - e_pt[0]
            dy = w_pt[1] - e_pt[1]
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            
            ux = dx / dist
            uy = dy / dist
            
            hand_len = max(15, min(30, int(dist * 0.35)))
            
            angles = [-0.6, -0.25, 0.0, 0.25, 0.5]
            finger_scales = [0.75, 0.95, 1.0, 0.95, 0.8]
            knuckles = []
            
            for i, a in enumerate(angles):
                cos_a = np.cos(a)
                sin_a = np.sin(a)
                rx = ux * cos_a - uy * sin_a
                ry = ux * sin_a + uy * cos_a
                
                k_len = hand_len * 0.45
                k_pt = (int(w_pt[0] + rx * k_len), int(w_pt[1] + ry * k_len))
                knuckles.append(k_pt)
                
                f_len = hand_len * finger_scales[i]
                jt = (int(k_pt[0] + rx * f_len * 0.4), int(k_pt[1] + ry * f_len * 0.4))
                tip = (int(k_pt[0] + rx * f_len * 0.8), int(k_pt[1] + ry * f_len * 0.8))
                
                # Draw wrist-to-knuckle (metacarpal bones)
                cv2.line(frame, (int(w_pt[0]), int(w_pt[1])), k_pt, (235, 235, 235), 1)
                # Draw knuckle-to-joint
                cv2.line(frame, k_pt, jt, (235, 235, 235), 1)
                # Draw joint-to-tip
                cv2.line(frame, jt, tip, (235, 235, 235), 1)
                
                # Draw micro joints
                cv2.circle(frame, k_pt, 1, (74, 163, 22), -1)
                cv2.circle(frame, jt, 1, (74, 163, 22), -1)
                cv2.circle(frame, tip, 1, (74, 163, 22), -1)
                
            # Connect the knuckles to form the palm arch mesh
            for i in range(len(knuckles) - 1):
                cv2.line(frame, knuckles[i], knuckles[i+1], (235, 235, 235), 1)

    def _draw_face_mesh(self, frame, nose, ear_l, ear_r, eye_l, eye_r, nose_conf, ear_l_conf, ear_r_conf, eye_l_conf, eye_r_conf):
        if nose_conf > 0.3 and ear_l_conf > 0.3 and ear_r_conf > 0.3:
            # Main keypoints as floats
            n_x, n_y = float(nose[0]), float(nose[1])
            el_x, el_y = float(ear_l[0]), float(ear_l[1])
            er_x, er_y = float(ear_r[0]), float(ear_r[1])
            
            # Use eyes if confidence is good, otherwise fallback to estimates
            if eye_l_conf > 0.3:
                eyel_x, eyel_y = float(eye_l[0]), float(eye_l[1])
            else:
                eyel_x = el_x * 0.4 + n_x * 0.6
                eyel_y = el_y * 0.3 + n_y * 0.7 - 20
                
            if eye_r_conf > 0.3:
                eyer_x, eyer_y = float(eye_r[0]), float(eye_r[1])
            else:
                eyer_x = er_x * 0.4 + n_x * 0.6
                eyer_y = er_y * 0.3 + n_y * 0.7 - 20

            # 1. Midpoints & Distance Scales
            mid_eye_x = (eyel_x + eyer_x) / 2.0
            mid_eye_y = (eyel_y + eyer_y) / 2.0
            
            # Vectors
            eye_dx = eyer_x - eyel_x
            eye_dy = eyer_y - eyel_y
            eye_dist = np.sqrt(eye_dx**2 + eye_dy**2) + 1e-6
            
            # Vertical face direction (mid-eye to nose vector)
            v_x = n_x - mid_eye_x
            v_y = n_y - mid_eye_y
            v_dist = np.sqrt(v_x**2 + v_y**2) + 1e-6
            
            # Normalised vertical vector
            nv_x = v_x / v_dist
            nv_y = v_y / v_dist
            
            # Normalised horizontal vector (perpendicular to vertical)
            nh_x = -nv_y
            nh_y = nv_x
            
            # 2. Define Procedural Face Grid Points (x, y)
            # Forehead / hairline
            f_mid = (int(mid_eye_x - nv_x * v_dist * 0.95), int(mid_eye_y - nv_y * v_dist * 0.95))
            f_l = (int(eyel_x - nv_x * v_dist * 0.90), int(eyel_y - nv_y * v_dist * 0.90))
            f_r = (int(eyer_x - nv_x * v_dist * 0.90), int(eyer_y - nv_y * v_dist * 0.90))
            
            # Eyebrows
            eb_l_mid = (int(eyel_x - nv_x * v_dist * 0.35), int(eyel_y - nv_y * v_dist * 0.35))
            eb_l_inner = (int(mid_eye_x - nv_x * v_dist * 0.38 - nh_x * eye_dist * 0.15), int(mid_eye_y - nv_y * v_dist * 0.38 - nh_y * eye_dist * 0.15))
            eb_l_outer = (int(eyel_x - nv_x * v_dist * 0.30 + nh_x * eye_dist * 0.20), int(eyel_y - nv_y * v_dist * 0.30 + nh_y * eye_dist * 0.20))
            
            eb_r_mid = (int(eyer_x - nv_x * v_dist * 0.35), int(eyer_y - nv_y * v_dist * 0.35))
            eb_r_inner = (int(mid_eye_x - nv_x * v_dist * 0.38 + nh_x * eye_dist * 0.15), int(mid_eye_y - nv_y * v_dist * 0.38 + nh_y * eye_dist * 0.15))
            eb_r_outer = (int(eyer_x - nv_x * v_dist * 0.30 - nh_x * eye_dist * 0.20), int(eyer_y - nv_y * v_dist * 0.30 - nh_y * eye_dist * 0.20))
            
            # Nose bridge structure
            bridge_top = (int(mid_eye_x), int(mid_eye_y))
            bridge_mid = (int(mid_eye_x + nv_x * v_dist * 0.5), int(mid_eye_y + nv_y * v_dist * 0.5))
            nose_tip = (int(n_x), int(n_y))
            nose_l = (int(n_x + nh_x * eye_dist * 0.22), int(n_y + nh_y * eye_dist * 0.22))
            nose_r = (int(n_x - nh_x * eye_dist * 0.22), int(n_y - nh_y * eye_dist * 0.22))
            
            # Eye orbits detail (diamonds around pupils)
            le_top = (int(eyel_x - nv_x * v_dist * 0.18), int(eyel_y - nv_y * v_dist * 0.18))
            le_bottom = (int(eyel_x + nv_x * v_dist * 0.18), int(eyel_y + nv_y * v_dist * 0.18))
            le_inner = (int(eyel_x - nh_x * eye_dist * 0.22), int(eyel_y - nh_y * eye_dist * 0.22))
            le_outer = (int(eyel_x + nh_x * eye_dist * 0.22), int(eyel_y + nh_y * eye_dist * 0.22))
            
            re_top = (int(eyer_x - nv_x * v_dist * 0.18), int(eyer_y - nv_y * v_dist * 0.18))
            re_bottom = (int(eyer_x + nv_x * v_dist * 0.18), int(eyer_y + nv_y * v_dist * 0.18))
            re_inner = (int(eyer_x + nh_x * eye_dist * 0.22), int(eyer_y + nh_y * eye_dist * 0.22))
            re_outer = (int(eyer_x - nh_x * eye_dist * 0.22), int(eyer_y - nh_y * eye_dist * 0.22))
            
            # Cheeks
            cheek_l = (int(eyel_x + nv_x * v_dist * 0.55 + nh_x * eye_dist * 0.12), int(eyel_y + nv_y * v_dist * 0.55 + nh_y * eye_dist * 0.12))
            cheek_r = (int(eyer_x + nv_x * v_dist * 0.55 - nh_x * eye_dist * 0.12), int(eyer_y + nv_y * v_dist * 0.55 - nh_y * eye_dist * 0.12))
            
            # Mouth / Lips
            m_mid_x = mid_eye_x + nv_x * v_dist * 1.55
            m_mid_y = mid_eye_y + nv_y * v_dist * 1.55
            mouth_l = (int(m_mid_x + nh_x * eye_dist * 0.35), int(m_mid_y + nh_y * eye_dist * 0.35))
            mouth_r = (int(m_mid_x - nh_x * eye_dist * 0.35), int(m_mid_y - nh_y * eye_dist * 0.35))
            mouth_top = (int(m_mid_x - nv_x * v_dist * 0.14), int(m_mid_y - nv_y * v_dist * 0.14))
            mouth_bot = (int(m_mid_x + nv_x * v_dist * 0.14), int(m_mid_y + nv_y * v_dist * 0.14))
            
            # Jawline / Chin outline
            chin = (int(mid_eye_x + nv_x * v_dist * 2.2), int(mid_eye_y + nv_y * v_dist * 2.2))
            chin_l = (int(chin[0] + nh_x * eye_dist * 0.22 - nv_x * v_dist * 0.08), int(chin[1] + nh_y * eye_dist * 0.22 - nv_y * v_dist * 0.08))
            chin_r = (int(chin[0] - nh_x * eye_dist * 0.22 - nv_x * v_dist * 0.08), int(chin[1] - nh_y * eye_dist * 0.22 - nv_y * v_dist * 0.08))
            
            jaw_l = (int(el_x + nv_x * v_dist * 0.5), int(el_y + nv_y * v_dist * 0.5))
            jaw_r = (int(er_x + nv_x * v_dist * 0.5), int(er_y + nv_y * v_dist * 0.5))
            
            # Soft Emerald BGR: (153, 211, 52)
            mesh_color = (153, 211, 52)
            
            if settings.YOLO_DEBUG_MODE:
                # ------------------- DEBUG MODE: FULL DENSE TRIANGULATION MESH -------------------
                connections = [
                    # Forehead hair line arch
                    (f_l, f_mid), (f_mid, f_r),
                    # Eyebrows arches
                    (eb_l_outer, eb_l_mid), (eb_l_mid, eb_l_inner),
                    (eb_r_inner, eb_r_mid), (eb_r_mid, eb_r_outer),
                    (eb_l_inner, eb_r_inner),
                    # Forehead to eyebrows
                    (f_l, eb_l_outer), (f_l, eb_l_mid), (f_mid, eb_l_inner), (f_mid, eb_r_inner), (f_r, eb_r_mid), (f_r, eb_r_outer),
                    # Forehead to temples/ears
                    (f_l, (int(el_x), int(el_y))), (f_r, (int(er_x), int(er_y))),
                    
                    # Eye socket loops (diamonds)
                    (le_top, le_outer), (le_outer, le_bottom), (le_bottom, le_inner), (le_inner, le_top),
                    (re_top, re_inner), (re_inner, re_bottom), (re_bottom, re_outer), (re_outer, re_top),
                    
                    # Eye center (pupil) connections to its sockets
                    ((int(eyel_x), int(eyel_y)), le_top), ((int(eyel_x), int(eyel_y)), le_outer), 
                    ((int(eyel_x), int(eyel_y)), le_bottom), ((int(eyel_x), int(eyel_y)), le_inner),
                    ((int(eyer_x), int(eyer_y)), re_top), ((int(eyer_x), int(eyer_y)), re_inner),
                    ((int(eyer_x), int(eyer_y)), re_bottom), ((int(eyer_x), int(eyer_y)), re_outer),
                    
                    # Brow to eye sockets
                    (eb_l_outer, le_outer), (eb_l_mid, le_top), (eb_l_inner, le_inner),
                    (eb_r_inner, re_inner), (eb_r_mid, re_top), (eb_r_outer, re_outer),
                    
                    # Nose structure
                    (bridge_top, bridge_mid), (bridge_mid, nose_tip),
                    (nose_l, nose_tip), (nose_tip, nose_r),
                    (bridge_mid, nose_l), (bridge_mid, nose_r),
                    # Nose bridge connections to brows
                    (bridge_top, eb_l_inner), (bridge_top, eb_r_inner),
                    # Nose to inner eye sockets
                    (bridge_mid, le_inner), (bridge_mid, re_inner),
                    
                    # Cheeks and outer connections
                    ((int(el_x), int(el_y)), le_outer), ((int(er_x), int(er_y)), re_outer),
                    ((int(el_x), int(el_y)), cheek_l), ((int(er_x), int(er_y)), cheek_r),
                    (le_bottom, cheek_l), (re_bottom, cheek_r),
                    (nose_l, cheek_l), (nose_r, cheek_r),
                    
                    # Mouth lip loop
                    (mouth_l, mouth_top), (mouth_top, mouth_r), (mouth_r, mouth_bot), (mouth_bot, mouth_l),
                    
                    # Nose to mouth connections
                    (nose_tip, mouth_top), (nose_l, mouth_l), (nose_r, mouth_r),
                    # Cheeks to mouth
                    (cheek_l, mouth_l), (cheek_r, mouth_r),
                    
                    # Jawline & Chin contour arch
                    ((int(el_x), int(el_y)), jaw_l), (jaw_l, chin_l), (chin_l, chin), (chin, chin_r), (chin_r, jaw_r), (jaw_r, (int(er_x), int(er_y))),
                    
                    # Chin to mouth connections
                    (chin_l, mouth_l), (chin, mouth_bot), (chin_r, mouth_r),
                    # Cheeks to jaw
                    (cheek_l, jaw_l), (cheek_r, jaw_r)
                ]
                for pt1, pt2 in connections:
                    cv2.line(frame, pt1, pt2, mesh_color, 1, lineType=cv2.LINE_AA)
                    
                all_mesh_pts = [
                    f_mid, f_l, f_r, eb_l_mid, eb_l_inner, eb_l_outer, eb_r_mid, eb_r_inner, eb_r_outer,
                    bridge_top, bridge_mid, nose_tip, nose_l, nose_r,
                    le_top, le_bottom, le_inner, le_outer, re_top, re_bottom, re_inner, re_outer,
                    cheek_l, cheek_r, mouth_l, mouth_r, mouth_top, mouth_bot,
                    chin, chin_l, chin_r, jaw_l, jaw_r
                ]
                for pt in all_mesh_pts:
                    cv2.circle(frame, pt, 2, mesh_color, -1, lineType=cv2.LINE_AA)
            else:
                # ------------------- PRODUCTION MODE: MINIMAL CLEAN CONTOURS -------------------
                # 1. Eyebrows
                eb_l = [eb_l_outer, eb_l_mid, eb_l_inner]
                eb_r = [eb_r_inner, eb_r_mid, eb_r_outer]
                for i in range(len(eb_l)-1):
                    cv2.line(frame, eb_l[i], eb_l[i+1], mesh_color, 1, lineType=cv2.LINE_AA)
                for i in range(len(eb_r)-1):
                    cv2.line(frame, eb_r[i], eb_r[i+1], mesh_color, 1, lineType=cv2.LINE_AA)
                
                # 2. Eye Orbits (Contours)
                eye_l_poly = np.array([le_top, le_outer, le_bottom, le_inner], dtype=np.int32)
                eye_r_poly = np.array([re_top, re_inner, re_bottom, re_outer], dtype=np.int32)
                cv2.polylines(frame, [eye_l_poly], True, mesh_color, 1, lineType=cv2.LINE_AA)
                cv2.polylines(frame, [eye_r_poly], True, mesh_color, 1, lineType=cv2.LINE_AA)
                
                # 3. Crosshair Pupils
                if eye_l_conf > 0.3:
                    cx, cy = int(eyel_x), int(eyel_y)
                    cv2.line(frame, (cx-3, cy), (cx+3, cy), mesh_color, 1, lineType=cv2.LINE_AA)
                    cv2.line(frame, (cx, cy-3), (cx, cy+3), mesh_color, 1, lineType=cv2.LINE_AA)
                if eye_r_conf > 0.3:
                    cx, cy = int(eyer_x), int(eyer_y)
                    cv2.line(frame, (cx-3, cy), (cx+3, cy), mesh_color, 1, lineType=cv2.LINE_AA)
                    cv2.line(frame, (cx, cy-3), (cx, cy+3), mesh_color, 1, lineType=cv2.LINE_AA)
                
                # 4. Nose Bridge and Nostrils
                cv2.line(frame, bridge_top, bridge_mid, mesh_color, 1, lineType=cv2.LINE_AA)
                cv2.line(frame, bridge_mid, nose_tip, mesh_color, 1, lineType=cv2.LINE_AA)
                cv2.line(frame, nose_l, nose_tip, mesh_color, 1, lineType=cv2.LINE_AA)
                cv2.line(frame, nose_tip, nose_r, mesh_color, 1, lineType=cv2.LINE_AA)
                
                # 5. Mouth Loop
                mouth_poly = np.array([mouth_l, mouth_top, mouth_r, mouth_bot], dtype=np.int32)
                cv2.polylines(frame, [mouth_poly], True, mesh_color, 1, lineType=cv2.LINE_AA)
                
                # 6. Jawline
                jaw_pts = [jaw_l, chin_l, chin, chin_r, jaw_r]
                for i in range(len(jaw_pts)-1):
                    cv2.line(frame, jaw_pts[i], jaw_pts[i+1], mesh_color, 1, lineType=cv2.LINE_AA)
                
                # 7. Draw 2px Circles on contour vertices only
                contour_pts = [
                    eb_l_outer, eb_l_mid, eb_l_inner, eb_r_inner, eb_r_mid, eb_r_outer,
                    le_top, le_outer, le_bottom, le_inner, re_top, re_inner, re_bottom, re_outer,
                    bridge_top, bridge_mid, nose_tip, nose_l, nose_r,
                    mouth_l, mouth_top, mouth_r, mouth_bot,
                    jaw_l, chin_l, chin, chin_r, jaw_r
                ]
                for pt in contour_pts:
                    cv2.circle(frame, pt, 2, mesh_color, -1, lineType=cv2.LINE_AA)


    def _generate_mock_frame(self):
        """
        Generates simulated frames to mimic a user at a desk with an advanced
        cybernetic-style thin skeleton and face mesh overlay.
        Cycles state rules for demo purposes:
          - 0-35s: Working (simulated typing / movement)
          - 35-70s: Idle (static sitting)
          - 70-90s: Absent (blank frame)
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (24, 20, 15)  # #0f172a (Steel dark background)

        # Draw office table layout
        cv2.rectangle(frame, (40, 390), (600, 480), (35, 42, 50), -1)
        cv2.rectangle(frame, (210, 280), (430, 390), (60, 60, 60), -1)
        cv2.rectangle(frame, (220, 290), (420, 375), (10, 10, 10), -1)
        cv2.rectangle(frame, (300, 390), (340, 420), (50, 50, 50), -1)

        cycle_time = int(time.time()) % 90
        
        detected = False
        confidence = 0.0
        landmarks = {}

        if cycle_time < 70:
            # User is present (WORKING or IDLE)
            detected = True
            t = time.time()
            
            # Setup active / idle parameters
            if cycle_time < 35:
                # WORKING state - active noise
                confidence = 0.94
                noise_x = int(np.sin(t * 8) * 4)
                noise_y = int(np.cos(t * 10) * 3)
                noise_hand_l = int(np.sin(t * 12) * 8)
                noise_hand_r = int(np.cos(t * 15) * 6)
                
                head_center = (320 + noise_x, 170 + noise_y)
                l_shoulder = (260, 250 + noise_y // 2)
                r_shoulder = (380, 250 + noise_y // 2)
                l_elbow = (240, 310 + noise_x)
                r_elbow = (400, 310 + noise_y)
                l_wrist = (270 + noise_hand_l, 350 + noise_hand_l // 2)
                r_wrist = (370 + noise_hand_r, 350 + noise_hand_r // 2)
                
                cv2.rectangle(frame, (240, 400), (400, 420), (74, 163, 22), 1)
                cv2.putText(frame, "SIMULATING ACTIVE WORK", (250, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (74, 163, 22), 1)
            else:
                # IDLE state - very static
                confidence = 0.90
                drift_x = int(np.sin(t * 0.4) * 0.4)
                drift_y = int(np.cos(t * 0.4) * 0.4)
                
                head_center = (330 + drift_x, 180 + drift_y)
                l_shoulder = (270, 260)
                r_shoulder = (390, 260)
                l_elbow = (240, 320)
                r_elbow = (420, 320)
                l_wrist = (230, 350)
                r_wrist = (410, 350)
                
                cv2.putText(frame, "SIMULATING STATIC IDLE", (230, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (11, 158, 245), 1)

            # 1. Draw background user body (very dark muted outlines)
            cv2.circle(frame, head_center, 28, (45, 40, 35), -1)
            cv2.line(frame, l_shoulder, r_shoulder, (45, 40, 35), 16)
            cv2.line(frame, l_shoulder, l_elbow, (45, 40, 35), 8)
            cv2.line(frame, l_elbow, l_wrist, (45, 40, 35), 8)
            cv2.line(frame, r_shoulder, r_elbow, (45, 40, 35), 8)
            cv2.line(frame, r_elbow, r_wrist, (45, 40, 35), 8)

            # 2. Draw high-fidelity thin tracking skeleton overlay
            # Thin off-white lines for skeleton limbs
            cv2.line(frame, l_shoulder, l_elbow, (235, 235, 235), 1)
            cv2.line(frame, l_elbow, l_wrist, (235, 235, 235), 1)
            cv2.line(frame, r_shoulder, r_elbow, (235, 235, 235), 1)
            cv2.line(frame, r_elbow, r_wrist, (235, 235, 235), 1)
            
            # Torso cage cross-bracing and midline
            s_mid_x = int((l_shoulder[0] + r_shoulder[0]) / 2.0)
            s_mid_y = int((l_shoulder[1] + r_shoulder[1]) / 2.0)
            # Neck line
            cv2.line(frame, head_center, (s_mid_x, s_mid_y), (235, 235, 235), 1)
            
            # Hips virtual midline
            h_mid_x = 320
            h_mid_y = 370
            # Spine line
            cv2.line(frame, (s_mid_x, s_mid_y), (h_mid_x, h_mid_y), (235, 235, 235), 1)
            # Virtual hip bones and side lines
            cv2.line(frame, (280, h_mid_y), (360, h_mid_y), (235, 235, 235), 1)
            cv2.line(frame, l_shoulder, (280, h_mid_y), (235, 235, 235), 1)
            cv2.line(frame, r_shoulder, (360, h_mid_y), (235, 235, 235), 1)
            
            # Torso cross braces
            cv2.line(frame, l_shoulder, (360, h_mid_y), (220, 220, 220), 1)
            cv2.line(frame, r_shoulder, (280, h_mid_y), (220, 220, 220), 1)

            # Thin green joint dots
            cv2.circle(frame, l_shoulder, 2, (74, 163, 22), -1)
            cv2.circle(frame, r_shoulder, 2, (74, 163, 22), -1)
            cv2.circle(frame, l_elbow, 2, (74, 163, 22), -1)
            cv2.circle(frame, r_elbow, 2, (74, 163, 22), -1)
            cv2.circle(frame, l_wrist, 2, (74, 163, 22), -1)
            cv2.circle(frame, r_wrist, 2, (74, 163, 22), -1)

            # 2.5 Draw mock hands and finger skeletons
            self._draw_hand_skeleton(frame, l_wrist, l_elbow, 1.0, 1.0)
            self._draw_hand_skeleton(frame, r_wrist, r_elbow, 1.0, 1.0)

            # 3. Draw thin cybernetic face mesh contours using mock coordinates
            mock_nose = (float(head_center[0]), float(head_center[1]))
            mock_ear_l = (float(head_center[0] - 22), float(head_center[1]))
            mock_ear_r = (float(head_center[0] + 22), float(head_center[1]))
            mock_eye_l = (float(head_center[0] - 8), float(head_center[1] - 2))
            mock_eye_r = (float(head_center[0] + 8), float(head_center[1] - 2))
            
            is_blinking_mock = (t % 4 < 0.25) # Blink for 0.25s every 4s
            if is_blinking_mock:
                # Closed eyes: draw mesh with 0.0 eye conf and draw flat indicator eye lines
                self._draw_face_mesh(
                    frame,
                    mock_nose, mock_ear_l, mock_ear_r, mock_eye_l, mock_eye_r,
                    1.0, 1.0, 1.0, 0.0, 0.0
                )
                cv2.line(frame, (int(mock_eye_l[0]-4), int(mock_eye_l[1])), (int(mock_eye_l[0]+4), int(mock_eye_l[1])), (20, 220, 240), 1)
                cv2.line(frame, (int(mock_eye_r[0]-4), int(mock_eye_r[1])), (int(mock_eye_r[0]+4), int(mock_eye_r[1])), (20, 220, 240), 1)
                cv2.putText(frame, "BLINK", (int(mock_nose[0])-16, int(mock_nose[1])-35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 220, 240), 1)
            else:
                self._draw_face_mesh(
                    frame,
                    mock_nose, mock_ear_l, mock_ear_r, mock_eye_l, mock_eye_r,
                    1.0, 1.0, 1.0, 1.0, 1.0
                )

            # Export landmarks for main loop displacement scoring
            landmarks[0] = (head_center[0]/640.0, head_center[1]/480.0)
            landmarks[11] = (l_shoulder[0]/640.0, l_shoulder[1]/480.0)
            landmarks[12] = (r_shoulder[0]/640.0, r_shoulder[1]/480.0)
            landmarks[15] = (l_wrist[0]/640.0, l_wrist[1]/480.0)
            landmarks[16] = (r_wrist[0]/640.0, r_wrist[1]/480.0)
            
        else:
            # ABSENT state
            detected = False
            confidence = 0.0
            landmarks = {}
            cv2.putText(frame, "EMPTY CHAIR (ABSENT)", (240, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (68, 68, 239), 1)

        return frame, detected, confidence, landmarks

# Create a single global monitor instance
cv_monitor = CVMonitor()
