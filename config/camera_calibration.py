import numpy as np
import cv2
import glob
import pickle

def calibrate_camera():
    """
    Calibrate the camera using a chessboard pattern.
    This will generate camera_params.pkl containing camera matrix and distortion coefficients.
    """
    # Chessboard dimensions (internal corners)
    chessboard_size = (8, 6)
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... etc.
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # Define real world spacing of chessboard squares in cm
    square_size = 1.7  # Standard chessboard is often 2.5cm per square
    objp = objp * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'c' to capture an image for calibration")
    print("Press 'q' to quit and calculate calibration")
    
    num_captures = 0
    min_captures = 10  # Minimum number of good captures for reliable calibration
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Display the current frame
        display = frame.copy()
        cv2.putText(display, f"Captures: {num_captures}/{min_captures}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Camera Calibration', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):  # Capture image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                # Refine corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw and display the corners
                cv2.drawChessboardCorners(display, chessboard_size, corners2, ret)
                cv2.imshow('Chessboard Detected', display)
                cv2.waitKey(500)  # Display for 500ms
                
                # Add object and image points
                objpoints.append(objp)
                imgpoints.append(corners2)
                
                num_captures += 1
                print(f"Image {num_captures} captured successfully")
            else:
                print("Chessboard not found in this image, try again")
        
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if num_captures < min_captures:
        print(f"Warning: Only {num_captures}/{min_captures} valid captures. Calibration may not be accurate.")
    
    if num_captures > 0:
        print("Calculating camera calibration...")
        # Get image size
        h, w = gray.shape
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None)
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        print(f"Total error: {mean_error/len(objpoints)}")
        
        # Save the calibration results
        calibration_data = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'rvecs': rvecs,
            'tvecs': tvecs
        }
        
        with open('camera_params.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print("Camera calibration parameters saved to camera_params.pkl")
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        return camera_matrix, dist_coeffs
    else:
        print("No images captured, calibration failed.")
        return None, None

def load_calibration():
    """
    Load camera calibration parameters from a file
    """
    try:
        with open('camera_params.pkl', 'rb') as f:
            data = pickle.load(f)
        print("Calibration loaded successfully")
        return data['camera_matrix'], data['dist_coeffs']
    except:
        print("Couldn't load calibration file, performing new calibration...")
        return calibrate_camera()

if __name__ == "__main__":
    camera_matrix, dist_coeffs = calibrate_camera()
