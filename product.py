#!/usr/bin/env python3
import time
from flask import Flask, request, jsonify, Response
from gpiozero import AngularServo, PhaseEnableMotor
from gpiozero.pins.pigpio import PiGPIOFactory
import sys
import os
import threading
import cv2 # OpenCV for image processing
import numpy as np
import pickle
import argparse
from picamera2 import Picamera2

# Parse command line arguments
parser = argparse.ArgumentParser(description='Robot Controller Server with optional preview window')
parser.add_argument('--preview', action='store_true', help='Show preview window with detected objects')
args = parser.parse_args()

# Initialize camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1296, 972)})
picam2.configure(camera_config)
picam2.start()

# Add frame sharing mechanism
frame_buffer = None
frame_lock = threading.Lock()
frame_available = threading.Event()

def frame_capture_loop():
    global frame_buffer, frame_available
    while True:
        try:
            frame = picam2.capture_array()
            with frame_lock:
                frame_buffer = frame.copy()
                frame_available.set()
        except Exception as e:
            print(f"Error capturing frame: {e}")
            time.sleep(0.1)

# Start frame capture thread
frame_capture_thread = threading.Thread(target=frame_capture_loop, daemon=True)
frame_capture_thread.start()

def generate_frames():
    while True:
        frame_available.wait()  # Wait for a new frame
        with frame_lock:
            if frame_buffer is not None:
                frame = frame_buffer.copy()
            else:
                continue
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# --- Configuration ---
# === ARM SERVO Configuration ===
BASE_PIN        = 17
SHOULDER_PIN    = 18
ELBOW_PIN       = 19
WRIST_PIN       = 12
GRIPPER_PIN     = 13
SERVO_MIN_PW = 0.0005
SERVO_MAX_PW = 0.0025
SERVO_MIN_ANGLE = -90
SERVO_MAX_ANGLE = 90
GRIPPER_MIN_ANGLE = 0
GRIPPER_MAX_ANGLE = 90

# === CHASSIS MOTOR Configuration ===
FL_MOTOR_PHASE = 5
FL_MOTOR_ENABLE = 6
FR_MOTOR_PHASE = 26
FR_MOTOR_ENABLE = 16
RL_MOTOR_PHASE = 20
RL_MOTOR_ENABLE = 21
RR_MOTOR_PHASE = 23
RR_MOTOR_ENABLE = 24
INVERT_RIGHT_MOTORS = True
INVERT_LEFT_MOTORS = False

try:
    factory = PiGPIOFactory()
    print("PiGPIOFactory initialized.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize PiGPIOFactory: {e}")
    sys.exit(1)

# --- Initialize Servos (ARM) ---
servos = {}
servo_limits = {}
try:
    base_servo = AngularServo(BASE_PIN, min_angle=SERVO_MIN_ANGLE, max_angle=SERVO_MAX_ANGLE, min_pulse_width=SERVO_MIN_PW, max_pulse_width=SERVO_MAX_PW, pin_factory=factory)
    shoulder_servo = AngularServo(SHOULDER_PIN, min_angle=SERVO_MIN_ANGLE, max_angle=SERVO_MAX_ANGLE, min_pulse_width=SERVO_MIN_PW, max_pulse_width=SERVO_MAX_PW, pin_factory=factory)
    elbow_servo = AngularServo(ELBOW_PIN, min_angle=SERVO_MIN_ANGLE, max_angle=SERVO_MAX_ANGLE, min_pulse_width=SERVO_MIN_PW, max_pulse_width=SERVO_MAX_PW, pin_factory=factory)
    wrist_servo = AngularServo(WRIST_PIN, min_angle=SERVO_MIN_ANGLE, max_angle=SERVO_MAX_ANGLE, min_pulse_width=SERVO_MIN_PW, max_pulse_width=SERVO_MAX_PW, pin_factory=factory)
    gripper_wrist_servo = AngularServo(GRIPPER_PIN, min_angle=GRIPPER_MIN_ANGLE, max_angle=GRIPPER_MAX_ANGLE, min_pulse_width=SERVO_MIN_PW, max_pulse_width=SERVO_MAX_PW, pin_factory=factory)
    servos = {"base": base_servo,"shoulder": shoulder_servo, "elbow": elbow_servo, "wrist": wrist_servo, "gripper_wrist": gripper_wrist_servo}
    servo_limits = {"base": (SERVO_MIN_ANGLE, SERVO_MAX_ANGLE),"shoulder": (SERVO_MIN_ANGLE, SERVO_MAX_ANGLE), "elbow": (SERVO_MIN_ANGLE, SERVO_MAX_ANGLE), "wrist": (SERVO_MIN_ANGLE, SERVO_MAX_ANGLE), "gripper_wrist": (GRIPPER_MIN_ANGLE, GRIPPER_MAX_ANGLE)}
    print("Arm Servos initialized successfully.")
except Exception as e:
    print(f"WARNING: Failed to initialize ARM servos: {e}")

# --- Initialize Motors (CHASSIS) ---
motors = {}
try:
    motor_fl = PhaseEnableMotor(phase=FL_MOTOR_PHASE, enable=FL_MOTOR_ENABLE, pwm=True, pin_factory=factory)
    motor_fr = PhaseEnableMotor(phase=FR_MOTOR_PHASE, enable=FR_MOTOR_ENABLE, pwm=True, pin_factory=factory)
    motor_rl = PhaseEnableMotor(phase=RL_MOTOR_PHASE, enable=RL_MOTOR_ENABLE, pwm=True, pin_factory=factory)
    motor_rr = PhaseEnableMotor(phase=RR_MOTOR_PHASE, enable=RR_MOTOR_ENABLE, pwm=True, pin_factory=factory)
    motors = {"fl": motor_fl, "fr": motor_fr, "rl": motor_rl, "rr": motor_rr}
    print("Chassis Motors initialized successfully.")
except Exception as e:
    print(f"WARNING: Failed to initialize CHASSIS motors: {e}")

# --- Cube Detection Configuration & Globals ---
DETECTION_CALIBRATION_FILE_PI = 'camera_params.pkl' # Ensure this file exists and is for 1296x972
DEFAULT_CUBE_SIZE_CM_PI = 3.0
DEFAULT_LOWER_RED1_PI = np.array([0, 120, 70])
DEFAULT_UPPER_RED1_PI = np.array([10, 255, 255])
DEFAULT_LOWER_RED2_PI = np.array([170, 120, 70])
DEFAULT_UPPER_RED2_PI = np.array([180, 255, 255])
DEFAULT_MORPH_KERNEL_SIZE_PI = (5, 5)
MIN_CONTOUR_AREA_PI = 150
MAX_ASPECT_RATIO_DEVIATION_PI = 0.35

#9 Target resolution for OpenCV VideoCapture
# This should match the resolution for which calibration was performed.
OPENCV_CAPTURE_WIDTH = 1296
OPENCV_CAPTURE_HEIGHT = 972

detector_thread = None
detection_running = False
latest_cube_coordinates_lock = threading.Lock()
latest_cube_coordinates = None
detector_camera_matrix = None
detector_dist_coeffs = None
loaded_calibration_resolution = None


def load_pi_camera_calibration(calibration_file):
    global detector_camera_matrix, detector_dist_coeffs, loaded_calibration_resolution
    if not os.path.exists(calibration_file):
        print(f"Error: Pi Calibration file not found at {calibration_file}")
        return False
    try:
        with open(calibration_file, 'rb') as f:
            data = pickle.load(f)
        detector_camera_matrix = data['camera_matrix']
        detector_dist_coeffs = data['dist_coeffs']
        loaded_calibration_resolution = data.get('calibration_resolution', (OPENCV_CAPTURE_WIDTH, OPENCV_CAPTURE_HEIGHT))
        print(f"Pi Calibration loaded from {calibration_file} for resolution {loaded_calibration_resolution}")

        # Check if loaded calibration matches the target OpenCV capture resolution
        if loaded_calibration_resolution != (OPENCV_CAPTURE_WIDTH, OPENCV_CAPTURE_HEIGHT):
            print(f"CRITICAL WARNING: Calibration resolution {loaded_calibration_resolution} from file "
                  f"does NOT match target OpenCV capture/processing resolution "
                  f"{(OPENCV_CAPTURE_WIDTH, OPENCV_CAPTURE_HEIGHT)}!")
            print("         This will lead to inaccurate pose estimation. Ensure calibration file matches target resolution.")
            # return False # Optionally, prevent startup if resolutions mismatch critically
        return True
    except Exception as e:
        print(f"Error loading Pi calibration file '{calibration_file}': {e}")
        return False

def get_red_mask_pi(hsv_image):
    mask1 = cv2.inRange(hsv_image, DEFAULT_LOWER_RED1_PI, DEFAULT_UPPER_RED1_PI)
    mask2 = cv2.inRange(hsv_image, DEFAULT_LOWER_RED2_PI, DEFAULT_UPPER_RED2_PI)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones(DEFAULT_MORPH_KERNEL_SIZE_PI, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def find_cube_contour_pi(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None; best_box_points = None; largest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA_PI: continue
        rect = cv2.minAreaRect(contour); box_points = cv2.boxPoints(rect)
        width_rect, height_rect = rect[1]
        if min(width_rect, height_rect) < 1e-5: continue
        aspect_ratio = max(width_rect, height_rect) / min(width_rect, height_rect)
        if abs(aspect_ratio - 1.0) > MAX_ASPECT_RATIO_DEVIATION_PI: continue
        if area > largest_area:
            largest_area = area; best_contour = contour; best_box_points = np.intp(box_points)
    return best_contour, best_box_points

def order_points(pts):
    # pts: 4x2 array
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def estimate_pose_pi(box_points_2d, cube_size_cm, camera_matrix_pi, dist_coeffs_pi):
    model_points_3d = np.array([[0,0,0], [cube_size_cm,0,0], [cube_size_cm,cube_size_cm,0], [0,cube_size_cm,0]], dtype=np.float32)
    image_points_2d = np.array(box_points_2d, dtype=np.float32)
    if image_points_2d.shape[0] != 4: return None, None
    image_points_2d = order_points(image_points_2d)
    try:
        success, rvec, tvec = cv2.solvePnP(model_points_3d, image_points_2d, camera_matrix_pi, dist_coeffs_pi)
        print("PnP solution found")
        return (rvec, tvec) if success else (None, None)
    except Exception as e: print(f"solvePnP error on Pi: {e}"); return None, None

def red_cube_detection_loop_pi():
    global detection_running, latest_cube_coordinates, detector_camera_matrix, detector_dist_coeffs

    if detector_camera_matrix is None or detector_dist_coeffs is None:
        if not load_pi_camera_calibration(DETECTION_CALIBRATION_FILE_PI):
            print("Cube detection cannot start: Pi camera calibration not loaded or resolution mismatch.")
            detection_running = False
            return

    print("Cube detection loop started on Pi (using shared camera)...")
    print("===========================================")
    print("Waiting for red cube detection...")
    print("===========================================")
    
    while detection_running:
        try:
            frame_available.wait()  # Wait for a new frame
            with frame_lock:
                if frame_buffer is not None:
                    frame_bgr = frame_buffer.copy()
                else:
                    continue

            # Frame is already BGR. Ensure calibration matches this frame's resolution.
            undistorted_frame = cv2.undistort(frame_bgr, detector_camera_matrix, detector_dist_coeffs)
            hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
            mask = get_red_mask_pi(hsv)
            _, cube_box_points = find_cube_contour_pi(mask)

            # Create output frame for display
            output_frame = undistorted_frame.copy()
            found_cube = False

            current_coords = None
            if cube_box_points is not None and len(cube_box_points) == 4:
                rvec, tvec = estimate_pose_pi(cube_box_points, DEFAULT_CUBE_SIZE_CM_PI,
                                             detector_camera_matrix, detector_dist_coeffs)
                if tvec is not None:
                    current_coords = [-tvec[0][0] - 1, tvec[2][0] + 6, -tvec[1][0] + 14]
                    found_cube = True
                    # Draw the detected cube outline
                    cv2.drawContours(output_frame, [cube_box_points], 0, (0, 255, 255), 2)  # Yellow outline
                    
                    # Draw pose axes
                    axis_length = DEFAULT_CUBE_SIZE_CM_PI
                    axis_points = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,-axis_length]]).reshape(-1,3)
                    imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, detector_camera_matrix, detector_dist_coeffs)
                    imgpts = np.intp(imgpts)
                    
                    origin = tuple(imgpts[0].ravel())
                    x_end = tuple(imgpts[1].ravel())
                    y_end = tuple(imgpts[2].ravel())
                    z_end = tuple(imgpts[3].ravel())
                    
                    cv2.line(output_frame, origin, x_end, (0,0,255), 3)  # X-axis (Red)
                    cv2.line(output_frame, origin, y_end, (0,255,0), 3)  # Y-axis (Green)
                    cv2.line(output_frame, origin, z_end, (255,0,0), 3)  # Z-axis (Blue)
                    
                    # Display position information
                    pos_text = f"Pos (Cam): X={tvec[0][0]:.1f} Y={tvec[1][0]:.1f} Z={tvec[2][0]:.1f} cm"
                    cv2.putText(output_frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Print detailed position information to terminal
                    print("\n===========================================")
                    print("RED CUBE DETECTED!")
                    print(f"Position relative to camera:")
                    print(f"  X: {tvec[0][0]:.2f} cm (right positive)")
                    print(f"  Y: {tvec[1][0]:.2f} cm (down positive)")
                    print(f"  Z: {tvec[2][0]:.2f} cm (forward positive)")
                    print("===========================================")

            if not found_cube:
                cv2.putText(output_frame, "No red cube detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Only show preview windows if --preview flag is set
            if args.preview:
                cv2.imshow('Cube Detection (Press q to quit)', output_frame)

                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    detection_running = False
                    break

            with latest_cube_coordinates_lock:
                latest_cube_coordinates = current_coords

        except cv2.error as e_cv:
            print(f"OpenCV Error in Pi detection loop: {e_cv}")
            with latest_cube_coordinates_lock: latest_cube_coordinates = None
        except Exception as e_gen:
            print(f"General Error in Pi detection loop: {e_gen}")
            with latest_cube_coordinates_lock: latest_cube_coordinates = None

        time.sleep(0.05)

    if picam2:
        picam2.stop()
        picam2.close()
    if args.preview:
        cv2.destroyAllWindows()
    with latest_cube_coordinates_lock: latest_cube_coordinates = None
    print("\n===========================================")
    print("Cube detection loop stopped on Pi (Picamera2).")
    print("===========================================")

# --- Flask App Endpoints ---
app = Flask(__name__)

@app.route('/set_angles', methods=['POST'])
def set_angles_route():
    if not servos: return jsonify({"status": "error", "message": "Arm Servos not initialized."}), 500
    data = request.get_json();
    if not data: return jsonify({"status": "error", "message": "No arm data."}), 400
    errors, success_moves = {}, {}
    for name, servo_obj in servos.items():
        if name in data:
            try:
                if name in ("shoulder", "wrist"):
                    angle = -int(data[name])
                else:
                    angle = int(data[name])
                min_lim, max_lim = servo_limits[name]
                clamped_angle = max(min_lim, min(angle, max_lim))
                if clamped_angle != angle: print(f"Warn: {name} ({angle}) clamped to {clamped_angle}")
                servo_obj.angle = clamped_angle; success_moves[name] = clamped_angle
            except Exception as e: errors[name] = str(e); print(f"Error {name}: {e}")
    if errors: return jsonify({"status": "partial_error", "moved": success_moves, "errors": errors}), 400
    return jsonify({"status": "success", "moved": success_moves}), 200

# --- Motor Control Helper Functions ---
def stop_motors():
    if not motors: return
    print("Stopping all chassis motors.")
    for motor_obj in motors.values():
        #motor_obj.stop()
        motor_obj.forward(1)

def move_forward(power):
    if not motors: return
    if INVERT_LEFT_MOTORS: motors["fl"].backward(power)
    else: motors["fl"].forward(power)
    if INVERT_RIGHT_MOTORS: motors["fr"].backward(power)
    else: motors["fr"].forward(power)
    if INVERT_LEFT_MOTORS: motors["rl"].backward(power)
    else: motors["rl"].forward(power)
    if INVERT_RIGHT_MOTORS: motors["rr"].backward(power)
    else: motors["rr"].forward(power)

def move_backward(power):
    if not motors: return
    if INVERT_LEFT_MOTORS: motors["fl"].forward(power)
    else: motors["fl"].backward(power)
    if INVERT_RIGHT_MOTORS: motors["fr"].forward(power)
    else: motors["fr"].backward(power)
    if INVERT_LEFT_MOTORS: motors["rl"].forward(power)
    else: motors["rl"].backward(power)
    if INVERT_RIGHT_MOTORS: motors["rr"].forward(power)
    else: motors["rr"].backward(power)

def move_left(power):
    if not motors: return
    if INVERT_LEFT_MOTORS: motors["fl"].forward(power)
    else: motors["fl"].backward(power)
    if INVERT_RIGHT_MOTORS: motors["fr"].backward(power)
    else: motors["fr"].forward(power)
    if INVERT_LEFT_MOTORS: motors["rl"].backward(power)
    else: motors["rl"].forward(power)
    if INVERT_RIGHT_MOTORS: motors["rr"].forward(power)
    else: motors["rr"].backward(power)

def move_right(power):
    if not motors: return
    if INVERT_LEFT_MOTORS: motors["fl"].backward(power)
    else: motors["fl"].forward(power)
    if INVERT_RIGHT_MOTORS: motors["fr"].forward(power)
    else: motors["fr"].backward(power)
    if INVERT_LEFT_MOTORS: motors["rl"].forward(power)
    else: motors["rl"].backward(power)
    if INVERT_RIGHT_MOTORS: motors["rr"].backward(power)
    else: motors["rr"].forward(power)

def turn_left(power):
    if not motors: return
    if INVERT_LEFT_MOTORS: motors["fl"].forward(power)
    else: motors["fl"].backward(power)
    if INVERT_RIGHT_MOTORS: motors["fr"].backward(power)
    else: motors["fr"].forward(power)
    if INVERT_LEFT_MOTORS: motors["rl"].forward(power)
    else: motors["rl"].backward(power)
    if INVERT_RIGHT_MOTORS: motors["rr"].backward(power)
    else: motors["rr"].forward(power)

def turn_right(power):
    if not motors: return
    if INVERT_LEFT_MOTORS: motors["fl"].backward(power)
    else: motors["fl"].forward(power)
    if INVERT_RIGHT_MOTORS: motors["fr"].forward(power)
    else: motors["fr"].backward(power)
    if INVERT_LEFT_MOTORS: motors["rl"].backward(power)
    else: motors["rl"].forward(power)
    if INVERT_RIGHT_MOTORS: motors["rr"].forward(power)
    else: motors["rr"].backward(power)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/move_chassis', methods=['POST'])
def move_chassis_route():
    if not motors: return jsonify({"status": "error", "message": "Chassis Motors not initialized."}), 500
    data = request.get_json()
    if not data or 'command' not in data:
        print("Warn: Chassis req missing command. Stopping.")
        move_forward(1.0)  # This stops the motors
        return jsonify({"status": "error", "message": "Invalid/missing chassis command."}), 400
    command = data['command'].lower()
    if 'speed' not in data:
        print(f"Warn: Chassis req for '{command}' missing speed. Stopping.")
        move_forward(1.0)  # This stops the motors
        return jsonify({"status": "error", "message": f"Missing speed for command: {command}"}), 400
    try:
        received_speed = float(data['speed'])
        received_speed = max(0.0, min(received_speed, 1.0))
        actual_power_for_movement = 1.0 - received_speed
    except:
        print(f"Warn: Invalid speed '{data['speed']}'. Stopping.")
        move_forward(1.0)  # This stops the motors
        return jsonify({"status": "error", "message": f"Invalid speed value: {data['speed']}"}), 400
    action_map = {"forward": move_forward, "backward": move_backward, "left": move_left,
                  "right": move_right, "turn_left": turn_left, "turn_right": turn_right}
    if command in action_map: 
        action_map[command](actual_power_for_movement)
    elif command == "stop": 
        move_forward(1.0)  # This stops the motors
    else:
        print(f"Warn: Unknown chassis cmd '{command}'. Stopping.")
        move_forward(1.0)  # This stops the motors
        return jsonify({"status": "error", "message": f"Unknown command: {command}"}), 400
    return jsonify({"status": "success", "command_executed": command, "speed_sent_by_gui": received_speed, "actual_power_used": actual_power_for_movement }), 200

@app.route('/start_cube_detection', methods=['POST'])
def start_cube_detection_route():
    global detection_running, detector_thread
    if detection_running:
        return jsonify({"status": "info", "message": "Cube detection already running."})
    
    # Ensure any existing thread is cleaned up
    if detector_thread and detector_thread.is_alive():
        detection_running = False
        detector_thread.join(timeout=2.0)
        detector_thread = None
    
    if detector_camera_matrix is None or detector_dist_coeffs is None:
        if not load_pi_camera_calibration(DETECTION_CALIBRATION_FILE_PI):
            return jsonify({"status": "error", "message": "Pi camera calibration failed. Cannot start detection."}), 500
    
    detection_running = True
    detector_thread = threading.Thread(target=red_cube_detection_loop_pi, daemon=True)
    detector_thread.start()
    print("Cube detection thread started.")
    return jsonify({"status": "success", "message": "Cube detection started."})

@app.route('/stop_cube_detection', methods=['POST'])
def stop_cube_detection_route():
    global detection_running, detector_thread
    if detection_running:
        detection_running = False
        if detector_thread and detector_thread.is_alive():
            detector_thread.join(timeout=2.0)  # Wait for thread to finish
        detector_thread = None
        print("Cube detection thread stopped.")
        return jsonify({"status": "success", "message": "Cube detection process stopped."})
    return jsonify({"status": "info", "message": "Cube detection not running."})

@app.route('/get_cube_coordinates', methods=['GET'])
def get_cube_coordinates_route():
    global latest_cube_coordinates
    with latest_cube_coordinates_lock: coords_to_send = latest_cube_coordinates
    if coords_to_send is not None: return jsonify({"status": "success", "coordinates": coords_to_send})
    return jsonify({"status": "not_found", "message": "Cube not detected or detection not active."})

@app.route('/test', methods=['GET'])
def test_connection():
    arm_status = "OK" if servos else "ERROR (Not Initialized)"
    chassis_status = "OK" if motors else "ERROR (Not Initialized)"
    return jsonify({"status":"success", "message":"Controller server is alive!", "arm_controller":arm_status, "chassis_controller":chassis_status})

if __name__ == '__main__':
    print("Starting Robot Controller Server (Arm, Chassis, Cube Detection with Picamera2)...")
    if args.preview:
        print("Preview window enabled - press 'q' to quit detection")
    if not load_pi_camera_calibration(DETECTION_CALIBRATION_FILE_PI):
        print("CRITICAL WARNING: Pi camera calibration FAILED to load. Detection may be inaccurate/fail.")
    else:
        print("Pi camera calibration loaded.")
    print(f"Arm Pins: Base={BASE_PIN}, etc.")
    print(f"Chassis Pins: FL={FL_MOTOR_PHASE}/{FL_MOTOR_ENABLE}, etc.")
    print(f"Motor Inversion: Left={INVERT_LEFT_MOTORS}, Right={INVERT_RIGHT_MOTORS}")
    print("Ensure pigpio daemon is running ('sudo systemctl start pigpiod')")
    move_forward(1.0)
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"ERROR starting Flask server: {e}")
    finally:
        print("\nShutting down server and cleaning up GPIO...")
        detection_running = False
        if detector_thread and detector_thread.is_alive():
            print("Waiting for detection thread...")
            detector_thread.join(timeout=2.0)
            if detector_thread.is_alive(): print("Detection thread did not finish.")
        if motors: move_forward(1.0); print("Chassis motors commanded to stop.")
        if servos:
            for s_name, servo_obj in servos.items():
                try: servo_obj.close()
                except Exception as e_s: print(f"Error closing servo {s_name}: {e_s}")
        if motors:
            for m_name, motor_obj in motors.items():
                try: motor_obj.close()
                except Exception as e_m: print(f"Error closing motor {m_name}: {e_m}")
        print("GPIO cleanup attempted.")
