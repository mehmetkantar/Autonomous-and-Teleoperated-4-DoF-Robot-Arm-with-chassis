import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.animation as animation
from math import sin, cos, atan2, sqrt, pi, acos

class RobotArm:
    """
    A class representing a 4DoF robot arm with a gripper.
    
    The robot arm has the following joints:
    - Base: Rotates around the z-axis
    - Shoulder: Rotates around the y-axis
    - Elbow: Rotates around the y-axis
    - Wrist: Rotates around the y-axis
    
    And a gripper that can open and close.
    """
    
    def __init__(self):
        """Initialize the robot arm with default dimensions and joint angles."""
        # Dimensions in cm
        self.base_height = 13.0
        self.shoulder_length = 15.0
        self.elbow_length = 15.0
        self.gripper_length = 4.0
        
        # Joint angles in radians
        self.base_angle = 0.0      # Rotation around z-axis
        self.shoulder_angle = 0.0  # Rotation around y-axis
        self.elbow_angle = 0.0     # Rotation around y-axis
        self.wrist_angle = 0.0     # Rotation around y-axis
        self.gripper_state = 0.0   # 0.0 (closed) to 1.0 (open)
        
        # Calibration offsets in radians
        self.base_offset = 0.0
        self.shoulder_offset = 1.57
        self.elbow_offset = 0.0
        self.wrist_offset = 0.0
        
        # End effector position
        self.end_effector = np.array([0.0, 0.0, 0.0])
        
        # Gripper properties
        self.gripper_width = 4.0  # Maximum width when fully open in cm
        
        self.base_min = np.radians(-90)
        self.base_max = np.radians(90)
        self.shoulder_min = np.radians(-90)
        self.shoulder_max = np.radians(90)
        self.elbow_min = np.radians(-90)
        self.elbow_max = np.radians(90)
        self.wrist_min = np.radians(-90)
        self.wrist_max = np.radians(90)
        
        # Initialize the arm position
        self.update_position()
    
    def _enforce_joint_limits(self, angles):
        base, shoulder, elbow, wrist = angles
        base = np.clip(base, self.base_min, self.base_max)
        shoulder = np.clip(shoulder, self.shoulder_min, self.shoulder_max)
        elbow = np.clip(elbow, self.elbow_min, self.elbow_max)
        wrist = np.clip(wrist, self.wrist_min, self.wrist_max)
        return [base, shoulder, elbow, wrist]
    
    def set_dimensions(self, base_height=None, shoulder_length=None, 
                      elbow_length=None, gripper_length=None):
        """
        Update the dimensions of the robot arm components.
        
        Parameters:
        -----------
        base_height : float, optional
            Height of the base in cm
        shoulder_length : float, optional
            Length of the shoulder link in cm
        elbow_length : float, optional
            Length of the elbow link in cm
        gripper_length : float, optional
            Length of the gripper in cm
        """
        if base_height is not None:
            self.base_height = base_height
        if shoulder_length is not None:
            self.shoulder_length = shoulder_length
        if elbow_length is not None:
            self.elbow_length = elbow_length
        if gripper_length is not None:
            self.gripper_length = gripper_length
        self.update_position()
    
    def set_offsets(self, base_offset=None, shoulder_offset=None, 
                   elbow_offset=None, wrist_offset=None):
        """
        Set calibration offsets for each joint.
        
        Parameters:
        -----------
        base_offset : float, optional
            Offset for the base joint in radians
        shoulder_offset : float, optional
            Offset for the shoulder joint in radians
        elbow_offset : float, optional
            Offset for the elbow joint in radians
        wrist_offset : float, optional
            Offset for the wrist joint in radians
        """
        if base_offset is not None:
            self.base_offset = base_offset
        if shoulder_offset is not None:
            self.shoulder_offset = shoulder_offset
        if elbow_offset is not None:
            self.elbow_offset = elbow_offset
        if wrist_offset is not None:
            self.wrist_offset = wrist_offset
        self.update_position()
    
    def set_joint_angles(self, base_angle=None, shoulder_angle=None, 
                        elbow_angle=None, wrist_angle=None, gripper_state=None):
        """
        Set the joint angles and update the arm position.
        
        Parameters:
        -----------
        base_angle : float, optional
            Angle of the base joint in radians
        shoulder_angle : float, optional
            Angle of the shoulder joint in radians
        elbow_angle : float, optional
            Angle of the elbow joint in radians
        wrist_angle : float, optional
            Angle of the wrist joint in radians
        gripper_state : float, optional
            State of the gripper from 0.0 (closed) to 1.0 (open)
        """
        if base_angle is not None:
            self.base_angle = base_angle
        if shoulder_angle is not None:
            self.shoulder_angle = shoulder_angle
        if elbow_angle is not None:
            self.elbow_angle = elbow_angle
        if wrist_angle is not None:
            self.wrist_angle = wrist_angle
        if gripper_state is not None:
            self.gripper_state = max(0.0, min(1.0, gripper_state))  # Clamp between 0 and 1
        self.update_position()
    
    def update_position(self):
        """
        Update the positions of all joints and the end effector based on current joint angles.
        
        Returns:
        --------
        numpy.ndarray
            The position of the end effector [x, y, z]
        """
        # Apply calibration offsets
        base_angle = self.base_angle + self.base_offset
        shoulder_angle = self.shoulder_angle + self.shoulder_offset
        elbow_angle = self.elbow_angle + self.elbow_offset
        wrist_angle = self.wrist_angle + self.wrist_offset
        
        # Base position (origin)
        self.base_pos = np.array([0, 0, 0])
        
        # Top of base position
        self.base_top_pos = np.array([0, 0, self.base_height])
        
        # Shoulder position (after base rotation)
        x_shoulder = 0
        y_shoulder = 0
        z_shoulder = self.base_height
        self.shoulder_pos = np.array([x_shoulder, y_shoulder, z_shoulder])
        
        # Elbow position (after shoulder rotation)
        x_elbow = self.shoulder_length * sin(base_angle) * cos(shoulder_angle)
        y_elbow = self.shoulder_length * cos(base_angle) * cos(shoulder_angle)
        z_elbow = self.base_height + self.shoulder_length * sin(shoulder_angle)
        self.elbow_pos = np.array([x_elbow, y_elbow, z_elbow])
        
        # Wrist position (after elbow rotation)
        x_wrist = x_elbow + self.elbow_length * sin(base_angle) * cos(shoulder_angle + elbow_angle)
        y_wrist = y_elbow + self.elbow_length * cos(base_angle) * cos(shoulder_angle + elbow_angle)
        z_wrist = z_elbow + self.elbow_length * sin(shoulder_angle + elbow_angle)
        self.wrist_pos = np.array([x_wrist, y_wrist, z_wrist])
        
        # End effector position (after wrist rotation)
        x_end = x_wrist + self.gripper_length * sin(base_angle) * cos(shoulder_angle + elbow_angle + wrist_angle)
        y_end = y_wrist + self.gripper_length * cos(base_angle) * cos(shoulder_angle + elbow_angle + wrist_angle)
        z_end = z_wrist + self.gripper_length * sin(shoulder_angle + elbow_angle + wrist_angle)
        self.end_effector = np.array([x_end, y_end, z_end])
        
        # Calculate gripper points for visualization
        self.calculate_gripper_points(base_angle, shoulder_angle, elbow_angle, wrist_angle)
        
        return self.end_effector
    
    def calculate_gripper_points(self, base_angle, shoulder_angle, elbow_angle, wrist_angle):
        """
        Calculate the points for visualizing the gripper.
        
        Parameters:
        -----------
        base_angle : float
            Angle of the base joint in radians
        shoulder_angle : float
            Angle of the shoulder joint in radians
        elbow_angle : float
            Angle of the elbow joint in radians
        wrist_angle : float
            Angle of the wrist joint in radians
        """
        # Calculate the direction vectors for the gripper
        # Forward direction (along the gripper)
        forward_x = sin(base_angle) * cos(shoulder_angle + elbow_angle + wrist_angle)
        forward_y = cos(base_angle) * cos(shoulder_angle + elbow_angle + wrist_angle)
        forward_z = sin(shoulder_angle + elbow_angle + wrist_angle)
        forward = np.array([forward_x, forward_y, forward_z])
        
        # Up direction (perpendicular to forward, in the vertical plane)
        up_x = sin(base_angle) * sin(shoulder_angle + elbow_angle + wrist_angle)
        up_y = cos(base_angle) * sin(shoulder_angle + elbow_angle + wrist_angle)
        up_z = -cos(shoulder_angle + elbow_angle + wrist_angle)
        up = np.array([up_x, up_y, up_z])
        
        # Right direction (perpendicular to forward and up)
        right_x = cos(base_angle)
        right_y = -sin(base_angle)
        right_z = 0
        right = np.array([right_x, right_y, right_z])
        
        # Calculate the gripper width based on the gripper state
        half_width = self.gripper_width * self.gripper_state / 2.0
        
        # Calculate the gripper points
        # Base of the gripper (at the wrist)
        self.gripper_base = self.wrist_pos.copy()
        
        # Left finger base
        self.left_finger_base = self.wrist_pos + right * half_width
        
        # Right finger base
        self.right_finger_base = self.wrist_pos - right * half_width
        
        # Left finger tip
        self.left_finger_tip = self.end_effector + right * half_width
        
        # Right finger tip
        self.right_finger_tip = self.end_effector - right * half_width
    
    def forward_kinematics(self, base_angle, shoulder_angle, elbow_angle, wrist_angle):
        """
        Calculate the end effector position given joint angles (forward kinematics).
        
        Parameters:
        -----------
        base_angle : float
            Angle of the base joint in radians
        shoulder_angle : float
            Angle of the shoulder joint in radians
        elbow_angle : float
            Angle of the elbow joint in radians
        wrist_angle : float
            Angle of the wrist joint in radians
            
        Returns:
        --------
        numpy.ndarray
            The position of the end effector [x, y, z]
        """
        # Save original angles
        original_angles = (self.base_angle, self.shoulder_angle, self.elbow_angle, self.wrist_angle)
        
        # Set the angles
        self.set_joint_angles(base_angle, shoulder_angle, elbow_angle, wrist_angle)
        
        # Get the end effector position
        result = self.end_effector.copy()
        
        # Restore original angles
        self.set_joint_angles(*original_angles)
        
        return result
    
    def inverse_kinematics(self, target_x, target_y, target_z, config=None, orientation=None, max_iterations=1000, error_threshold=0.1, return_all=False):
        """
        Calculate joint angles to reach a target position (inverse kinematics).
        
        Parameters:
        -----------
        target_x : float
            Target x-coordinate in cm
        target_y : float
            Target y-coordinate in cm
        target_z : float
            Target z-coordinate in cm
        config : str, optional
            Desired arm configuration: 'elbow_up', 'elbow_down', or None (try both)
        orientation : float, original_state
            Desired end effector orientation (wrist angle in radians)
        max_iterations : int, optional
            Maximum number of iterations for solution refinement
        error_threshold : float, optional
            Acceptable error threshold in cm
        return_all : bool, optional
            If True, return all valid solutions found
            
        Returns:
        --------
        if return_all is False:
            tuple: (success, [base_angle, shoulder_angle, elbow_angle, wrist_angle])
                success is a boolean indicating whether a valid solution was found
        if return_all is True:
            tuple: (success, [[base_angle, shoulder_angle, elbow_angle, wrist_angle], ...])
                success is a boolean indicating whether any valid solution was found
                The list contains all valid solutions, ordered by error (best first)
        """
        # Save original state
        original_state = (self.base_angle, self.shoulder_angle, self.elbow_angle, self.wrist_angle, self.gripper_state)
        
        # Store target
        target = np.array([target_x, target_y, target_z])
        all_solutions = []
        
        # Maximum reach
        max_reach = self.shoulder_length + self.elbow_length + self.gripper_length
        
        # Base reachability check
        distance_xy = sqrt(target_x**2 + target_y**2)
        distance_xyz = sqrt(distance_xy**2 + (target_z - self.base_height)**2)
        
        if distance_xyz > max_reach:
            print(f"Target out of reach. Maximum reach: {max_reach:.2f}cm, Target distance: {distance_xyz:.2f}cm")
            return False, [self.base_angle, self.shoulder_angle, self.elbow_angle, self.wrist_angle]
        
        # Calculate base angle
        if abs(target_x) < 1e-6 and abs(target_y) < 1e-6:
            # Target is directly above or below
            if return_all:
                # Try multiple base angles
                base_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
            else:
                # Keep current angle
                base_angles = [self.base_angle]
        else:
            # Calculate the angle in the XY plane
            base_angle = atan2(target_x, target_y)
            base_angles = [base_angle]
        
        # Determine configurations to try
        if config == 'elbow_up':
            configs = ['elbow_up']
        elif config == 'elbow_down':
            configs = ['elbow_down']
        else:
            configs = ['elbow_up', 'elbow_down']
        
        # Try each base angle and configuration
        for base_angle in base_angles:
            for config in configs:
                # Initialize solution
                solution = None
                
                # Initial guess based on configuration
                if config == 'elbow_up':
                    # Start with arm pointing upward
                    shoulder_angle = np.pi/4  # 45 degrees up
                    elbow_angle = -np.pi/2    # -90 degrees (bend upward)
                else:
                    # Start with arm pointing downward
                    shoulder_angle = -np.pi/4  # 45 degrees down
                    elbow_angle = np.pi/2     # 90 degrees (bend downward)
                
                wrist_angle = 0  # Start with neutral wrist
                
                # Save initial guess
                initial_guess = [base_angle, shoulder_angle, elbow_angle, wrist_angle]
                
                # Numerical optimization using gradient descent
                current_angles = initial_guess.copy()
                best_error = float('inf')
                best_angles = current_angles.copy()
                
                for iteration in range(max_iterations):
                    # Calculate current position
                    current_position = self.forward_kinematics(*current_angles)
                    
                    # Calculate error
                    error_vector = target - current_position
                    error = np.linalg.norm(error_vector)
                    
                    # Save best solution
                    if error < best_error:
                        best_error = error
                        best_angles = current_angles.copy()
                    
                    # Check if we've reached the error threshold
                    if error < error_threshold:
                        break
                    
                    # Calculate numerical Jacobian
                    # For each joint, calculate how much the end effector moves when the joint changes slightly
                    jacobian = np.zeros((3, 4))  # 3 dimensions, 4 joints
                    eps = 0.001  # Small angle change for numerical differentiation
                    
                    for i in range(4):  # For each joint
                        # Create a small change in angle
                        delta_angles = current_angles.copy()
                        delta_angles[i] += eps
                        
                        # Calculate new position
                        new_position = self.forward_kinematics(*delta_angles)
                        
                        # Calculate column of Jacobian (change in position / change in angle)
                        jacobian[:, i] = (new_position - current_position) / eps
                    
                    # Calculate pseudo-inverse of Jacobian
                    # This allows us to map from desired position change to joint angle changes
                    try:
                        # Use pseudo-inverse for better numerical stability
                        jacobian_pinv = np.linalg.pinv(jacobian)
                        
                        # Calculate joint angle updates
                        delta_angles = jacobian_pinv.dot(error_vector)
                        
                        # Limit step size to prevent overshooting
                        step_size = min(0.1, error)
                        delta_angles = delta_angles * step_size / np.linalg.norm(delta_angles) if np.linalg.norm(delta_angles) > 0 else delta_angles
                        
                        # Update angles
                        new_angles = [
                            current_angles[0] + delta_angles[0],
                            current_angles[1] + delta_angles[1],
                            current_angles[2] + delta_angles[2],
                            current_angles[3] + delta_angles[3]
                        ]
                        
                        # Check if this is a valid configuration
                        if config == 'elbow_up' and new_angles[2] > 0:
                            # Enforce elbow up (negative elbow angle)
                            new_angles[2] = -abs(new_angles[2])
                        elif config == 'elbow_down' and new_angles[2] < 0:
                            # Enforce elbow down (positive elbow angle)
                            new_angles[2] = abs(new_angles[2])
                        
                        current_angles = new_angles
                        
                    except np.linalg.LinAlgError:
                        # If pseudo-inverse fails, use a simple update strategy
                        if error_vector[2] > 0:  # Target is higher than current position
                            current_angles[1] += 0.05  # Move shoulder up
                        else:
                            current_angles[1] -= 0.05  # Move shoulder down
                            
                        # Keep trying to move in target direction
                        xy_direction = np.array([target_x, target_y]) - np.array([current_position[0], current_position[1]])
                        if np.linalg.norm(xy_direction) > 0:
                            xy_direction = xy_direction / np.linalg.norm(xy_direction)
                            current_angles[0] += 0.05 * xy_direction[0]  # Adjust base
                
                # Final position check
                final_position = self.forward_kinematics(*best_angles)
                error = np.linalg.norm(target - final_position)
                
                # If orientation is specified, adjust wrist angle
                if orientation is not None:
                    # Calculate current orientation (just the sum of angles)
                    current_orientation = best_angles[1] + best_angles[2] + best_angles[3]
                    
                    # Calculate required wrist angle adjustment
                    wrist_adjustment = orientation - current_orientation
                    
                    # Update wrist angle
                    best_angles[3] += wrist_adjustment
                    
                    # Verify orientation adjustment didn't significantly affect position
                    check_position = self.forward_kinematics(*best_angles)
                    new_error = np.linalg.norm(target - check_position)
                    
                    if new_error > 2 * error:  # If error doubled
                        # Revert orientation adjustment
                        best_angles[3] -= wrist_adjustment
                
                # Accept solution if error is reasonable
                if error < error_threshold * 10:  # Allow 10x the threshold for acceptance
                    # Store solution with its error
                    all_solutions.append({
                        'angles': best_angles,
                        'error': error,
                        'config': config
                    })
        
        # Restore original state
        self.set_joint_angles(*original_state[:-1], original_state[-1])
        
        # Check if we found any solutions
        if not all_solutions:
            return False, [self.base_angle, self.shoulder_angle, self.elbow_angle, self.wrist_angle]
        
        # Sort solutions by error
        all_solutions.sort(key=lambda x: x['error'])
        
        if return_all:
            return True, [sol['angles'] for sol in all_solutions]
        else:
            # Use the best solution
            best_solution = all_solutions[0]
            
            # Apply the best solution
            self.set_joint_angles(*best_solution['angles'])
            
            print(f"Found solution ({best_solution['config']}) with error: {best_solution['error']:.4f}cm")
            
            return True, best_solution['angles']
    
    def get_arm_positions(self):
        """
        Return the positions of all joints for plotting.
        
        Returns:
        --------
        tuple
            (x_points, y_points, z_points) where each is a list of coordinates
        """
        x_points = [self.base_pos[0], self.shoulder_pos[0], self.elbow_pos[0], 
                   self.wrist_pos[0], self.end_effector[0]]
        y_points = [self.base_pos[1], self.shoulder_pos[1], self.elbow_pos[1], 
                   self.wrist_pos[1], self.end_effector[1]]
        z_points = [self.base_pos[2], self.shoulder_pos[2], self.elbow_pos[2], 
                   self.wrist_pos[2], self.end_effector[2]]
        return x_points, y_points, z_points
    
    def get_gripper_positions(self):
        """
        Return the positions of the gripper for plotting.
        
        Returns:
        --------
        tuple
            (left_x, left_y, left_z, right_x, right_y, right_z) where each is a list of coordinates
        """
        # Left finger
        left_x = [self.wrist_pos[0], self.left_finger_base[0], self.left_finger_tip[0]]
        left_y = [self.wrist_pos[1], self.left_finger_base[1], self.left_finger_tip[1]]
        left_z = [self.wrist_pos[2], self.left_finger_base[2], self.left_finger_tip[2]]
        
        # Right finger
        right_x = [self.wrist_pos[0], self.right_finger_base[0], self.right_finger_tip[0]]
        right_y = [self.wrist_pos[1], self.right_finger_base[1], self.right_finger_tip[1]]
        right_z = [self.wrist_pos[2], self.right_finger_base[2], self.right_finger_tip[2]]
        
        return left_x, left_y, left_z, right_x, right_y, right_z
