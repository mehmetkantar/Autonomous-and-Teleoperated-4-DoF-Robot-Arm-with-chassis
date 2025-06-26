"""
Comparison Simulation - Ideal vs. Real Robot Arm

This simulation shows a side-by-side comparison between:
1. Ideal inverse kinematics solution (left) - Using exact calculated angles
2. Real robot arm execution (right) - Using rounded integer angles to simulate motor limitations

The simulation shows both arms following the same path but with different precision.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QComboBox, QLabel, QPushButton, 
                           QGridLayout, QGroupBox, QSlider, QTabWidget,QLineEdit)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import from the robot arm and path planner files
from robot_arm_improved import RobotArm
from path_planner import PathPlanner

class ComparisonSimulationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup the main window
        self.setWindowTitle("Robot Arm Comparison Simulation: Ideal vs. Real")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create ideal and real robot arm instances
        self.ideal_robot = RobotArm()
        self.real_robot = RobotArm()
        
        # Animation parameters
        self.animation_path = []
        self.ideal_solutions = []  # Exact angle solutions
        self.real_solutions = []   # Rounded angle solutions
        self.animation_running = False
        self.animation_index = 0
        self.angle_value_font_style = "font-size: 12pt;"
        # Setup the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Setup controls panel
        self.setup_control_panel(main_layout)
        
        # Setup the 3D plots - side by side
        self.setup_3d_plots(main_layout)
        
        # Initialize the plots
        self.update_plots()
    
    def setup_control_panel(self, main_layout):
        """Setup the control panel for the simulation with multiple input methods"""
        control_panel = QGroupBox("Simulation Controls")
        control_layout = QGridLayout()
        control_panel.setLayout(control_layout)
        
        # Path type selection
        control_layout.addWidget(QLabel("Path Type:"), 0, 0)
        self.path_type_combo = QComboBox()
        self.path_type_combo.addItems(["Straight Line", "Half Circle Up", "Half Circle Down", "Half Figure-8"])
        control_layout.addWidget(self.path_type_combo, 0, 1)
        
        # Arm configuration selection
        control_layout.addWidget(QLabel("Arm Configuration:"), 0, 2)
        self.config_combo = QComboBox()
        self.config_combo.addItems(["Auto (Best)", "Elbow Up", "Elbow Down"])
        control_layout.addWidget(self.config_combo, 0, 3)
        
        # Create tabs for different input methods
        self.input_tab_widget = QTabWidget()
        
        # Tab 1: Coordinate Input
        coords_tab = QWidget()
        coords_layout = QGridLayout(coords_tab)
        
        # Start point coordinates
        coords_layout.addWidget(QLabel("Start Point:"), 0, 0)
        coords_layout.addWidget(QLabel("X:"), 0, 1)
        self.start_x_input = QLineEdit("0")
        self.start_x_input.setMaximumWidth(60)
        coords_layout.addWidget(self.start_x_input, 0, 2)
        coords_layout.addWidget(QLabel("cm"), 0, 3)
        
        coords_layout.addWidget(QLabel("Y:"), 0, 4)
        self.start_y_input = QLineEdit("0")
        self.start_y_input.setMaximumWidth(60)
        coords_layout.addWidget(self.start_y_input, 0, 5)
        coords_layout.addWidget(QLabel("cm"), 0, 6)
        
        coords_layout.addWidget(QLabel("Z:"), 0, 7)
        self.start_z_input = QLineEdit("15")
        self.start_z_input.setMaximumWidth(60)
        coords_layout.addWidget(self.start_z_input, 0, 8)
        coords_layout.addWidget(QLabel("cm"), 0, 9)
        
        # End point coordinates
        coords_layout.addWidget(QLabel("End Point:"), 1, 0)
        coords_layout.addWidget(QLabel("X:"), 1, 1)
        self.end_x_input = QLineEdit("20")
        self.end_x_input.setMaximumWidth(60)
        coords_layout.addWidget(self.end_x_input, 1, 2)
        coords_layout.addWidget(QLabel("cm"), 1, 3)
        
        coords_layout.addWidget(QLabel("Y:"), 1, 4)
        self.end_y_input = QLineEdit("20")
        self.end_y_input.setMaximumWidth(60)
        coords_layout.addWidget(self.end_y_input, 1, 5)
        coords_layout.addWidget(QLabel("cm"), 1, 6)
        
        coords_layout.addWidget(QLabel("Z:"), 1, 7)
        self.end_z_input = QLineEdit("15")
        self.end_z_input.setMaximumWidth(60)
        coords_layout.addWidget(self.end_z_input, 1, 8)
        coords_layout.addWidget(QLabel("cm"), 1, 9)
        
        # Tab 2: Motor Angle Input
        angles_tab = QWidget()
        angles_layout = QGridLayout(angles_tab)
        
        # Start motor angles
        angles_layout.addWidget(QLabel("Start Angles:"), 0, 0)
        angles_layout.addWidget(QLabel("Base:"), 0, 1)
        self.start_base_input = QLineEdit("0")
        self.start_base_input.setMaximumWidth(60)
        angles_layout.addWidget(self.start_base_input, 0, 2)
        angles_layout.addWidget(QLabel("°"), 0, 3)
        
        angles_layout.addWidget(QLabel("Shoulder:"), 0, 4)
        self.start_shoulder_input = QLineEdit("0")
        self.start_shoulder_input.setMaximumWidth(60)
        angles_layout.addWidget(self.start_shoulder_input, 0, 5)
        angles_layout.addWidget(QLabel("°"), 0, 6)
        
        angles_layout.addWidget(QLabel("Elbow:"), 0, 7)
        self.start_elbow_input = QLineEdit("0")
        self.start_elbow_input.setMaximumWidth(60)
        angles_layout.addWidget(self.start_elbow_input, 0, 8)
        angles_layout.addWidget(QLabel("°"), 0, 9)
        
        angles_layout.addWidget(QLabel("Wrist:"), 0, 10)
        self.start_wrist_input = QLineEdit("0")
        self.start_wrist_input.setMaximumWidth(60)
        angles_layout.addWidget(self.start_wrist_input, 0, 11)
        angles_layout.addWidget(QLabel("°"), 0, 12)
        
        # End motor angles
        angles_layout.addWidget(QLabel("End Angles:"), 1, 0)
        angles_layout.addWidget(QLabel("Base:"), 1, 1)
        self.end_base_input = QLineEdit("45")
        self.end_base_input.setMaximumWidth(60)
        angles_layout.addWidget(self.end_base_input, 1, 2)
        angles_layout.addWidget(QLabel("°"), 1, 3)
        
        angles_layout.addWidget(QLabel("Shoulder:"), 1, 4)
        self.end_shoulder_input = QLineEdit("45")
        self.end_shoulder_input.setMaximumWidth(60)
        angles_layout.addWidget(self.end_shoulder_input, 1, 5)
        angles_layout.addWidget(QLabel("°"), 1, 6)
        
        angles_layout.addWidget(QLabel("Elbow:"), 1, 7)
        self.end_elbow_input = QLineEdit("45")
        self.end_elbow_input.setMaximumWidth(60)
        angles_layout.addWidget(self.end_elbow_input, 1, 8)
        angles_layout.addWidget(QLabel("°"), 1, 9)
        
        angles_layout.addWidget(QLabel("Wrist:"), 1, 10)
        self.end_wrist_input = QLineEdit("0")
        self.end_wrist_input.setMaximumWidth(60)
        angles_layout.addWidget(self.end_wrist_input, 1, 11)
        angles_layout.addWidget(QLabel("°"), 1, 12)
        
        # Calculate buttons
        calc_layout = QHBoxLayout()
        
        # Button to calculate position from angles
        self.calc_pos_btn = QPushButton("Calculate Positions from Angles")
        self.calc_pos_btn.clicked.connect(self.calculate_positions_from_angles)
        calc_layout.addWidget(self.calc_pos_btn)
        
        # Button to calculate angles from position
        self.calc_angles_btn = QPushButton("Calculate Angles from Positions")
        self.calc_angles_btn.clicked.connect(self.calculate_angles_from_positions)
        calc_layout.addWidget(self.calc_angles_btn)
        
        angles_layout.addLayout(calc_layout, 2, 0, 1, 13)
        
        # Add tabs to tab widget
        self.input_tab_widget.addTab(coords_tab, "Coordinate Input")
        self.input_tab_widget.addTab(angles_tab, "Motor Angle Input")
        
        # Add tab widget to control layout
        control_layout.addWidget(self.input_tab_widget, 1, 0, 1, 4)
        
        # Animation speed
        control_layout.addWidget(QLabel("Animation Speed:"), 2, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        control_layout.addWidget(self.speed_slider, 2, 1, 1, 3)
        
        # Animation control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)
        button_layout.addWidget(self.stop_button)
        
        self.reset_button = QPushButton("Reset Simulation")
        self.reset_button.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.reset_button)
        
        # Add buttons to control layout
        control_layout.addLayout(button_layout, 3, 0, 1, 4)
        
        # Status display
        control_layout.addWidget(QLabel("Status:"), 4, 0)
        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label, 4, 1, 1, 3)
        
        # Add to main layout with appropriate height for all controls
        control_panel.setMaximumHeight(300)
        main_layout.addWidget(control_panel)
    
    def setup_3d_plots(self, main_layout):
        """Setup the side by side 3D plots for comparison with synchronized camera views"""
        # Create a widget to hold the two plots side by side
        plots_widget = QWidget()
        plots_layout = QHBoxLayout(plots_widget)
        
        # Left plot - Ideal kinematics
        self.ideal_figure = plt.figure(figsize=(8, 8))
        self.ideal_canvas = FigureCanvas(self.ideal_figure)
        self.ideal_ax = self.ideal_figure.add_subplot(111, projection='3d')
        self.ideal_ax.set_title('Ideal Inverse Kinematics\n(Exact Angles)')
        self.setup_axis(self.ideal_ax)
        
        # Initialize ideal robot plot elements
        self.ideal_arm_line, = self.ideal_ax.plot([], [], [], 'o-', linewidth=2, markersize=6, color='blue')
        self.ideal_end_effector, = self.ideal_ax.plot([], [], [], 'ro', markersize=8)
        self.ideal_gripper_left, = self.ideal_ax.plot([], [], [], 'g-', linewidth=1.5)
        self.ideal_gripper_right, = self.ideal_ax.plot([], [], [], 'g-', linewidth=1.5)
        self.ideal_target, = self.ideal_ax.plot([], [], [], 'yo', markersize=8)
        self.ideal_path_line, = self.ideal_ax.plot([], [], [], 'c-', linewidth=1, alpha=0.5)
        
        # Right plot - Real robot with integer angles
        self.real_figure = plt.figure(figsize=(8, 8))
        self.real_canvas = FigureCanvas(self.real_figure)
        self.real_ax = self.real_figure.add_subplot(111, projection='3d')
        self.real_ax.set_title('Real Robot Arm\n(Integer Angles Only)')
        self.setup_axis(self.real_ax)
        
        # Initialize real robot plot elements
        self.real_arm_line, = self.real_ax.plot([], [], [], 'o-', linewidth=2, markersize=6, color='red')
        self.real_end_effector, = self.real_ax.plot([], [], [], 'ro', markersize=8)
        self.real_gripper_left, = self.real_ax.plot([], [], [], 'g-', linewidth=1.5)
        self.real_gripper_right, = self.real_ax.plot([], [], [], 'g-', linewidth=1.5)
        self.real_target, = self.real_ax.plot([], [], [], 'yo', markersize=8)
        self.real_path_line, = self.real_ax.plot([], [], [], 'c-', linewidth=1, alpha=0.5)
        
        # Connect events to synchronize views
        self.ideal_canvas.mpl_connect('motion_notify_event', self.on_ideal_view_change)
        self.real_canvas.mpl_connect('motion_notify_event', self.on_real_view_change)
        
        # Add plots to the layout
        left_plot_group = QGroupBox("Ideal Robot Arm")
        left_plot_layout = QVBoxLayout()
        left_plot_layout.addWidget(self.ideal_canvas)
        left_plot_group.setLayout(left_plot_layout)
        
        right_plot_group = QGroupBox("Real Robot Arm (Integer Angles)")
        right_plot_layout = QVBoxLayout()
        right_plot_layout.addWidget(self.real_canvas)
        right_plot_group.setLayout(right_plot_layout)
        
        plots_layout.addWidget(left_plot_group)
        plots_layout.addWidget(right_plot_group)
        
        # Add the plots widget to the main layout
        main_layout.addWidget(plots_widget)
        
        # Add angle display panel at the bottom
        self.setup_angle_display(main_layout)
    
    def setup_angle_display(self, main_layout):
        """Setup a panel to display current angles for both robots"""
        angle_panel = QGroupBox("Current Joint Angles")
        angle_layout = QGridLayout()
        angle_panel.setLayout(angle_layout)

        # Headers
        angle_layout.addWidget(QLabel(""), 0, 0)
        
        # Modify these lines to set a larger font size for the headers
        header_font_style = "font-size: 14pt; font-weight: bold;"

        # Corrected lines for headers:
        ideal_header_label = QLabel("Ideal (degrees)")
        ideal_header_label.setStyleSheet(header_font_style)
        angle_layout.addWidget(ideal_header_label, 0, 1)

        real_header_label = QLabel("Real (degrees)")
        real_header_label.setStyleSheet(header_font_style)
        angle_layout.addWidget(real_header_label, 0, 2)

        diff_header_label = QLabel("Difference")
        diff_header_label.setStyleSheet(header_font_style)
        angle_layout.addWidget(diff_header_label, 0, 3)

        # Define a common style for the angle values
        angle_value_font_style = "font-size: 12pt;" # You can change 12pt to your desired size

        # Base angle
        angle_layout.addWidget(QLabel("Base Angle:"), 1, 0)
        self.ideal_base_label = QLabel("0.00°")
        self.ideal_base_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.ideal_base_label, 1, 1)
        self.real_base_label = QLabel("0°")
        self.real_base_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.real_base_label, 1, 2)
        self.diff_base_label = QLabel("0.00°")
        self.diff_base_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.diff_base_label, 1, 3)

        # Shoulder angle
        angle_layout.addWidget(QLabel("Shoulder Angle:"), 2, 0)
        self.ideal_shoulder_label = QLabel("0.00°")
        self.ideal_shoulder_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.ideal_shoulder_label, 2, 1)
        self.real_shoulder_label = QLabel("0°")
        self.real_shoulder_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.real_shoulder_label, 2, 2)
        self.diff_shoulder_label = QLabel("0.00°")
        self.diff_shoulder_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.diff_shoulder_label, 2, 3)

        # Elbow angle
        angle_layout.addWidget(QLabel("Elbow Angle:"), 3, 0)
        self.ideal_elbow_label = QLabel("0.00°")
        self.ideal_elbow_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.ideal_elbow_label, 3, 1)
        self.real_elbow_label = QLabel("0°")
        self.real_elbow_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.real_elbow_label, 3, 2)
        self.diff_elbow_label = QLabel("0.00°")
        self.diff_elbow_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.diff_elbow_label, 3, 3)

        # Wrist angle
        angle_layout.addWidget(QLabel("Wrist Angle:"), 4, 0)
        self.ideal_wrist_label = QLabel("0.00°")
        self.ideal_wrist_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.ideal_wrist_label, 4, 1)
        self.real_wrist_label = QLabel("0°")
        self.real_wrist_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.real_wrist_label, 4, 2)
        self.diff_wrist_label = QLabel("0.00°")
        self.diff_wrist_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.diff_wrist_label, 4, 3)

        # Position error
        angle_layout.addWidget(QLabel("Position Error:"), 5, 0)
        self.ideal_error_label = QLabel("0.00 cm")
        self.ideal_error_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.ideal_error_label, 5, 1)
        self.real_error_label = QLabel("0.00 cm")
        self.real_error_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.real_error_label, 5, 2)
        self.diff_error_label = QLabel("0.00 cm")
        self.diff_error_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.diff_error_label, 5, 3)

        # Current target position
        angle_layout.addWidget(QLabel("Target:"), 6, 0)
        self.target_pos_label = QLabel("(0.00, 0.00, 0.00) cm")
        self.target_pos_label.setStyleSheet(angle_value_font_style)
        angle_layout.addWidget(self.target_pos_label, 6, 1, 1, 3)

        # Set a fixed height for the angle panel
        #angle_panel.setMaximumHeight(200)
        main_layout.addWidget(angle_panel)  
        
    def on_ideal_view_change(self, event):
        """Handle view changes in the ideal plot and sync with real plot"""
        if event.inaxes == self.ideal_ax:
            # Only sync if this is a rotation event (button press)
            if hasattr(event, 'button') and event.button is not None:
                # Get current view angles
                elev, azim = self.ideal_ax.elev, self.ideal_ax.azim
                
                # Update the other plot to match
                self.real_ax.view_init(elev=elev, azim=azim)
                self.real_canvas.draw_idle()
    
    def on_real_view_change(self, event):
        """Handle view changes in the real plot and sync with ideal plot"""
        if event.inaxes == self.real_ax:
            # Only sync if this is a rotation event (button press)
            if hasattr(event, 'button') and event.button is not None:
                # Get current view angles
                elev, azim = self.real_ax.elev, self.real_ax.azim
                
                # Update the other plot to match
                self.ideal_ax.view_init(elev=elev, azim=azim)
                self.ideal_canvas.draw_idle()
    
    def setup_axis(self, ax):
        """Setup the 3D axis with common settings"""
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        
        # Set axis limits based on robot reach
        max_reach = self.ideal_robot.base_height + self.ideal_robot.shoulder_length + self.ideal_robot.elbow_length + self.ideal_robot.gripper_length
        ax.set_xlim(-max_reach/2, max_reach)
        ax.set_ylim(-max_reach/2, max_reach)
        ax.set_zlim(0, max_reach)
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set a consistent viewing angle
        ax.view_init(elev=30, azim=45)
    
    def update_plots(self):
        """Update both robot arm plots"""
        # Update ideal robot plot
        self.update_robot_plot(
            self.ideal_robot, 
            self.ideal_arm_line, 
            self.ideal_end_effector, 
            self.ideal_gripper_left, 
            self.ideal_gripper_right
        )
        
        # Update real robot plot
        self.update_robot_plot(
            self.real_robot, 
            self.real_arm_line, 
            self.real_end_effector, 
            self.real_gripper_left, 
            self.real_gripper_right
        )
        
        # Redraw canvases
        self.ideal_canvas.draw()
        self.real_canvas.draw()
    
    def calculate_positions_from_angles(self):
        """Calculate and display the positions corresponding to the input angles"""
        try:
            # Get start angles
            start_base = np.radians(float(self.start_base_input.text()))
            start_shoulder = np.radians(float(self.start_shoulder_input.text()))
            start_elbow = np.radians(float(self.start_elbow_input.text()))
            start_wrist = np.radians(float(self.start_wrist_input.text()))
            
            # Get end angles
            end_base = np.radians(float(self.end_base_input.text()))
            end_shoulder = np.radians(float(self.end_shoulder_input.text()))
            end_elbow = np.radians(float(self.end_elbow_input.text()))
            end_wrist = np.radians(float(self.end_wrist_input.text()))
            
            # Calculate positions using forward kinematics
            start_pos = self.ideal_robot.forward_kinematics(start_base, start_shoulder, start_elbow, start_wrist)
            end_pos = self.ideal_robot.forward_kinematics(end_base, end_shoulder, end_elbow, end_wrist)
            
            # Update position inputs
            self.start_x_input.setText(f"{start_pos[0]:.2f}")
            self.start_y_input.setText(f"{start_pos[1]:.2f}")
            self.start_z_input.setText(f"{start_pos[2]:.2f}")
            
            self.end_x_input.setText(f"{end_pos[0]:.2f}")
            self.end_y_input.setText(f"{end_pos[1]:.2f}")
            self.end_z_input.setText(f"{end_pos[2]:.2f}")
            
            # Switch to coordinates tab to show the results
            self.input_tab_widget.setCurrentIndex(0)
            
            # Update status
            self.status_label.setText("Positions calculated from angles successfully.")
            
        except ValueError as e:
            self.status_label.setText(f"Error: {str(e)}")
    
    def calculate_angles_from_positions(self):
        """Calculate and display the angles corresponding to the input positions"""
        try:
            # Get start position
            start_x = float(self.start_x_input.text())
            start_y = float(self.start_y_input.text())
            start_z = float(self.start_z_input.text())
            
            # Get end position
            end_x = float(self.end_x_input.text())
            end_y = float(self.end_y_input.text())
            end_z = float(self.end_z_input.text())
            
            # Get selected configuration
            config_option = self.config_combo.currentText()
            if config_option == "Elbow Up":
                config = "elbow_up"
            elif config_option == "Elbow Down":
                config = "elbow_down"
            else:
                config = None  # Auto (best)
            
            # Calculate angles using inverse kinematics
            success_start, start_angles = self.ideal_robot.inverse_kinematics(
                start_x, start_y, start_z,
                config=config,
                error_threshold=0.1,
                max_iterations=1000
            )
            
            success_end, end_angles = self.ideal_robot.inverse_kinematics(
                end_x, end_y, end_z,
                config=config,
                error_threshold=0.1,
                max_iterations=1000
            )
            
            if success_start and success_end:
                # Update angle inputs (convert to degrees)
                self.start_base_input.setText(f"{np.degrees(start_angles[0]):.2f}")
                self.start_shoulder_input.setText(f"{np.degrees(start_angles[1]):.2f}")
                self.start_elbow_input.setText(f"{np.degrees(start_angles[2]):.2f}")
                self.start_wrist_input.setText(f"{np.degrees(start_angles[3]):.2f}")
                
                self.end_base_input.setText(f"{np.degrees(end_angles[0]):.2f}")
                self.end_shoulder_input.setText(f"{np.degrees(end_angles[1]):.2f}")
                self.end_elbow_input.setText(f"{np.degrees(end_angles[2]):.2f}")
                self.end_wrist_input.setText(f"{np.degrees(end_angles[3]):.2f}")
                
                # Switch to angles tab to show the results
                self.input_tab_widget.setCurrentIndex(1)
                
                # Update status
                self.status_label.setText("Angles calculated from positions successfully.")
            else:
                if not success_start:
                    self.status_label.setText("Error: Could not calculate angles for start position.")
                else:
                    self.status_label.setText("Error: Could not calculate angles for end position.")
                
        except ValueError as e:
            self.status_label.setText(f"Error: {str(e)}")
    
    def update_robot_plot(self, robot, arm_line, end_effector, gripper_left, gripper_right):
        """Update a single robot arm plot"""
        # Get arm positions
        x_points, y_points, z_points = robot.get_arm_positions()
        
        # Update arm line
        arm_line.set_data(x_points, y_points)
        arm_line.set_3d_properties(z_points)
        
        # Update end effector
        end_effector.set_data([robot.end_effector[0]], [robot.end_effector[1]])
        end_effector.set_3d_properties([robot.end_effector[2]])
        
        # Update gripper
        left_x, left_y, left_z, right_x, right_y, right_z = robot.get_gripper_positions()
        
        gripper_left.set_data(left_x, left_y)
        gripper_left.set_3d_properties(left_z)
        
        gripper_right.set_data(right_x, right_y)
        gripper_right.set_3d_properties(right_z)
    
    def generate_comparison_path(self):
        """Generate path and calculate both ideal and rounded solutions"""
        # Clear previous paths and solutions
        self.animation_path = []
        self.ideal_solutions = []
        self.real_solutions = []
        
        try:
            # Get start and end points from coordinate input fields
            start_point = (
                float(self.start_x_input.text()),
                float(self.start_y_input.text()),
                float(self.start_z_input.text())
            )
            
            end_point = (
                float(self.end_x_input.text()),
                float(self.end_y_input.text()),
                float(self.end_z_input.text())
            )
        except ValueError:
            self.status_label.setText("Error: Invalid coordinate input. Please enter valid numbers.")
            return False
        
        # Number of points in the path
        num_points = 20
        
        # Get selected path type
        path_type = self.path_type_combo.currentText()
        
        print(f"Generating path: {path_type}")
        print(f"Start point: ({start_point[0]:.2f}, {start_point[1]:.2f}, {start_point[2]:.2f})")
        print(f"End point: ({end_point[0]:.2f}, {end_point[1]:.2f}, {end_point[2]:.2f})")
        
        # Use path planner to generate path points
        if path_type == "Straight Line":
            self.animation_path = PathPlanner.generate_path('straight', start_point, end_point, num_points)
        elif path_type == "Half Circle Up":
            self.animation_path = PathPlanner.generate_path('half_circle_up', start_point, end_point, num_points)
        elif path_type == "Half Circle Down":
            self.animation_path = PathPlanner.generate_path('half_circle_down', start_point, end_point, num_points)
        elif path_type == "Half Figure-8":
            self.animation_path = PathPlanner.generate_path('half_figure8', start_point, end_point, num_points)
        
        print(f"Generated path with {len(self.animation_path)} points")
        
        # Draw the path on both plots
        path_x = [p[0] for p in self.animation_path]
        path_y = [p[1] for p in self.animation_path]
        path_z = [p[2] for p in self.animation_path]
        
        self.ideal_path_line.set_data(path_x, path_y)
        self.ideal_path_line.set_3d_properties(path_z)
        
        self.real_path_line.set_data(path_x, path_y)
        self.real_path_line.set_3d_properties(path_z)
        
        # Calculate IK solutions for all path points
        self.status_label.setText("Calculating solutions for path...")
        print("Calculating inverse kinematics solutions for each path point...")
        
        # Get selected configuration
        config_option = self.config_combo.currentText()
        if config_option == "Elbow Up":
            config = "elbow_up"
        elif config_option == "Elbow Down":
            config = "elbow_down"
        else:
            config = None  # Auto (best)
        
        print(f"Using arm configuration: {config if config else 'Auto (Best)'}")
        
        # Pre-calculate solutions for both robots
        solution_count = 0
        for i, point in enumerate(self.animation_path):
            x, y, z = point
            
            # Calculate ideal solution
            success, ideal_solution = self.ideal_robot.inverse_kinematics(
                x, y, z, 
                config=config,
                error_threshold=0.1,
                max_iterations=1000
            )
            
            if success:
                solution_count += 1
                self.ideal_solutions.append(ideal_solution)
                
                # Create rounded solution for real robot
                base_angle, shoulder_angle, elbow_angle, wrist_angle = ideal_solution
                
                # Round to integers (degrees then back to radians)
                base_deg = round(np.degrees(base_angle))
                shoulder_deg = round(np.degrees(shoulder_angle))
                elbow_deg = round(np.degrees(elbow_angle))
                wrist_deg = round(np.degrees(wrist_angle))
                
                real_solution = (
                    np.radians(base_deg),
                    np.radians(shoulder_deg),
                    np.radians(elbow_deg),
                    np.radians(wrist_deg)
                )
                
                self.real_solutions.append(real_solution)
                
                # Print progress every 5 points or for the first and last point
                if i % 5 == 0 or i == 0 or i == len(self.animation_path) - 1:
                    print(f"Point {i+1}/{len(self.animation_path)}: ({x:.2f}, {y:.2f}, {z:.2f}) - Solution found")
                    print(f"  Ideal Angles: Base={np.degrees(base_angle):.2f}°, Shoulder={np.degrees(shoulder_angle):.2f}°, Elbow={np.degrees(elbow_angle):.2f}°")
                    print(f"  Real Angles:  Base={base_deg}°, Shoulder={shoulder_deg}°, Elbow={elbow_deg}°")
            else:
                # If no solution found, append None to both lists
                self.ideal_solutions.append(None)
                self.real_solutions.append(None)
                print(f"Point {i+1}/{len(self.animation_path)}: ({x:.2f}, {y:.2f}, {z:.2f}) - NO SOLUTION FOUND")
        
        # Update the status
        if solution_count == len(self.animation_path):
            self.status_label.setText(f"Path generated with {len(self.animation_path)} points. Ready to simulate.")
            print(f"All {solution_count} points have valid solutions. Ready to simulate.")
            return True
        else:
            self.status_label.setText(f"Some points couldn't be reached ({solution_count}/{len(self.animation_path)}). Try another path.")
            print(f"WARNING: Only {solution_count} out of {len(self.animation_path)} points have valid solutions.")
            return False
        
        
    def check_point_reachable(self, point):
        """Check if a point is within the robot's reach"""
        x, y, z = point
        
        # Calculate distance from origin (base of robot)
        distance_xy = np.sqrt(x**2 + y**2)
        distance_z = z - self.ideal_robot.base_height
        
        # Calculate Euclidean distance
        distance = np.sqrt(distance_xy**2 + distance_z**2)
        
        # Check if within robot's reach
        max_reach = self.ideal_robot.shoulder_length + self.ideal_robot.elbow_length + self.ideal_robot.gripper_length
        
        return distance <= max_reach
    
    def start_simulation(self):
        """Start the animation comparing ideal and real robots"""
        if self.animation_running:
            return
        
        try:
            # Check if start and end points are reachable
            start_point = (
                float(self.start_x_input.text()),
                float(self.start_y_input.text()),
                float(self.start_z_input.text())
            )
            
            end_point = (
                float(self.end_x_input.text()),
                float(self.end_y_input.text()),
                float(self.end_z_input.text())
            )
        except ValueError:
            self.status_label.setText("Error: Invalid coordinate input. Please enter valid numbers.")
            return
        
        if not self.check_point_reachable(start_point):
            self.status_label.setText("Error: Start point is out of reach!")
            return
            
        if not self.check_point_reachable(end_point):
            self.status_label.setText("Error: End point is out of reach!")
            return
        
        # Update status
        self.status_label.setText("Generating path and calculating solutions...")
        
        # Generate the path and solutions
        if not self.generate_comparison_path():
            return
        
        # Start the animation
        self.animation_running = True
        self.animation_index = 0
        
        # Show the start and end target points on both plots
        start_point = self.animation_path[0]
        end_point = self.animation_path[-1]
        
        self.ideal_target.set_data([end_point[0]], [end_point[1]])
        self.ideal_target.set_3d_properties([end_point[2]])
        
        self.real_target.set_data([end_point[0]], [end_point[1]])
        self.real_target.set_3d_properties([end_point[2]])
        
        # Update the display to show path before animation starts
        self.ideal_canvas.draw()
        self.real_canvas.draw()
        
        # Update status
        self.status_label.setText("Animation running...")
        
        # Start animation
        self.perform_animation_step()
    
    def stop_simulation(self):
        """Stop the animation"""
        self.animation_running = False
        self.status_label.setText("Animation stopped.")
    
    def reset_simulation(self):
        """Reset the animation to initial state"""
        self.animation_running = False
        self.animation_index = 0
        
        try:
            # Get start position from input fields
            start_x = float(self.start_x_input.text())
            start_y = float(self.start_y_input.text())
            start_z = float(self.start_z_input.text())
        except ValueError:
            # If input is invalid, use default values
            start_x, start_y, start_z = 0, 0, 15
            self.start_x_input.setText("0")
            self.start_y_input.setText("0")
            self.start_z_input.setText("15")
            self.status_label.setText("Warning: Invalid coordinate input. Using default start position.")
        
        # Try to solve inverse kinematics for the start position
        try:
            # Get selected configuration
            config_option = self.config_combo.currentText()
            if config_option == "Elbow Up":
                config = "elbow_up"
            elif config_option == "Elbow Down":
                config = "elbow_down"
            else:
                config = None  # Auto (best)
                
            # Solve IK for ideal robot
            success_ideal, ideal_solution = self.ideal_robot.inverse_kinematics(
                start_x, start_y, start_z,
                config=config,
                error_threshold=0.1,
                max_iterations=1000
            )
            
            # Round the angles to integers for real robot
            if success_ideal:
                base_angle, shoulder_angle, elbow_angle, wrist_angle = ideal_solution
                
                # Convert to degrees, round, and convert back to radians
                base_deg = round(np.degrees(base_angle))
                shoulder_deg = round(np.degrees(shoulder_angle))
                elbow_deg = round(np.degrees(elbow_angle))
                wrist_deg = round(np.degrees(wrist_angle))
                
                real_solution = (
                    np.radians(base_deg),
                    np.radians(shoulder_deg),
                    np.radians(elbow_deg),
                    np.radians(wrist_deg)
                )
                
                # Set robot positions
                self.ideal_robot.set_joint_angles(*ideal_solution, 0.5)
                self.real_robot.set_joint_angles(*real_solution, 0.5)
                
                # Update angle display
                self.update_angle_display(ideal_solution, real_solution, (start_x, start_y, start_z))
            else:
                # Fall back to home position if IK fails
                self.ideal_robot.set_joint_angles(0, 0, 0, 0, 0.5)
                self.real_robot.set_joint_angles(0, 0, 0, 0, 0.5)
                self.update_angle_display(None, None, None)
                self.status_label.setText("Warning: Could not solve IK for start position, using home position.")
        except Exception as e:
            # If there's an error, reset to home position
            self.ideal_robot.set_joint_angles(0, 0, 0, 0, 0.5)
            self.real_robot.set_joint_angles(0, 0, 0, 0, 0.5)
            self.update_angle_display(None, None, None)
            self.status_label.setText(f"Error during reset: {str(e)}")
        
        # Clear paths
        self.ideal_path_line.set_data([], [])
        self.ideal_path_line.set_3d_properties([])
        self.real_path_line.set_data([], [])
        self.real_path_line.set_3d_properties([])
        
        # Clear targets
        self.ideal_target.set_data([], [])
        self.ideal_target.set_3d_properties([])
        self.real_target.set_data([], [])
        self.real_target.set_3d_properties([])
        
        # Update plots
        self.update_plots()
        
        # Update status (unless there was an error)
        if "Error" not in self.status_label.text() and "Warning" not in self.status_label.text():
            self.status_label.setText("Simulation reset to start position.")
    def perform_animation_step(self):
        """Perform a single step of the animation"""
        if not self.animation_running or not self.animation_path:
            return
        
        # Get the current point and solutions
        point = self.animation_path[self.animation_index]
        ideal_solution = self.ideal_solutions[self.animation_index]
        real_solution = self.real_solutions[self.animation_index]
        
        x, y, z = point
        
        # Update target position label
        self.target_pos_label.setText(f"({x:.2f}, {y:.2f}, {z:.2f}) cm")
        
        # Set ideal robot position
        if ideal_solution:
            self.ideal_robot.set_joint_angles(*ideal_solution, 0.5)
        
        # Set real robot position with rounded angles
        if real_solution:
            self.real_robot.set_joint_angles(*real_solution, 0.5)
        
        # Update angles display
        self.update_angle_display(ideal_solution, real_solution, point)
        
        # Print current animation step information
        print(f"Step {self.animation_index}/{len(self.animation_path)-1} - Target: ({x:.2f}, {y:.2f}, {z:.2f}) cm")
        if ideal_solution and real_solution:
            ideal_base_deg = np.degrees(ideal_solution[0])
            ideal_shoulder_deg = np.degrees(ideal_solution[1])
            ideal_elbow_deg = np.degrees(ideal_solution[2])
            ideal_wrist_deg = np.degrees(ideal_solution[3])
            
            real_base_deg = np.degrees(real_solution[0])
            real_shoulder_deg = np.degrees(real_solution[1])
            real_elbow_deg = np.degrees(real_solution[2])
            real_wrist_deg = np.degrees(real_solution[3])
            
            # Calculate position errors
            ideal_target = np.array([x, y, z])
            ideal_error = np.linalg.norm(ideal_target - self.ideal_robot.end_effector)
            real_error = np.linalg.norm(ideal_target - self.real_robot.end_effector)
            
            print(f"  Ideal Angles: Base={ideal_base_deg:.2f}°, Shoulder={ideal_shoulder_deg:.2f}°, Elbow={ideal_elbow_deg:.2f}°, Wrist={ideal_wrist_deg:.2f}°")
            print(f"  Real Angles:  Base={real_base_deg:.0f}°, Shoulder={real_shoulder_deg:.0f}°, Elbow={real_elbow_deg:.0f}°, Wrist={real_wrist_deg:.0f}°")
            print(f"  Position Error: Ideal={ideal_error:.4f}cm, Real={real_error:.4f}cm, Diff={(real_error-ideal_error):.4f}cm")
            print("---------------------------------------------")
        
        # Update the plots
        self.update_plots()
        
        # Move to the next point
        self.animation_index += 1
        
        # If we reached the end of the path, stop the animation
        if self.animation_index >= len(self.animation_path):
            self.animation_running = False
            self.status_label.setText("Animation completed.")
            print("Animation completed.")
            return
        
        # Schedule the next step if animation is still running
        if self.animation_running:
            # Get animation speed (inversely proportional to delay)
            speed = self.speed_slider.value()
            delay = int(1000 / speed)  # 100-1000ms
            
            # Use QTimer for the next step
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(delay, self.perform_animation_step)
            
        
    def update_angle_display(self, ideal_solution, real_solution, target_point):
        """Update the angle display panel with current angles and differences"""
        if ideal_solution and real_solution and target_point:
            # Get angles from solutions
            ideal_base, ideal_shoulder, ideal_elbow, ideal_wrist = ideal_solution
            real_base, real_shoulder, real_elbow, real_wrist = real_solution
            
            # Convert to degrees for display
            ideal_base_deg = np.degrees(ideal_base)
            ideal_shoulder_deg = np.degrees(ideal_shoulder)
            ideal_elbow_deg = np.degrees(ideal_elbow)
            ideal_wrist_deg = np.degrees(ideal_wrist)
            
            real_base_deg = np.degrees(real_base)
            real_shoulder_deg = np.degrees(real_shoulder)
            real_elbow_deg = np.degrees(real_elbow)
            real_wrist_deg = np.degrees(real_wrist)
            
            # Calculate differences
            base_diff = abs(ideal_base_deg - real_base_deg)
            shoulder_diff = abs(ideal_shoulder_deg - real_shoulder_deg)
            elbow_diff = abs(ideal_elbow_deg - real_elbow_deg)
            wrist_diff = abs(ideal_wrist_deg - real_wrist_deg)
            
            # Calculate position errors
            x, y, z = target_point
            ideal_target = np.array([x, y, z])
            
            ideal_error = np.linalg.norm(ideal_target - self.ideal_robot.end_effector)
            real_error = np.linalg.norm(ideal_target - self.real_robot.end_effector)
            error_diff = abs(real_error - ideal_error)
            
            # Update angle labels
            self.ideal_base_label.setText(f"{ideal_base_deg:.2f}°")
            self.real_base_label.setText(f"{real_base_deg:.0f}°")
            self.diff_base_label.setText(f"{base_diff:.2f}°")
            
            self.ideal_shoulder_label.setText(f"{ideal_shoulder_deg:.2f}°")
            self.real_shoulder_label.setText(f"{real_shoulder_deg:.0f}°")
            self.diff_shoulder_label.setText(f"{shoulder_diff:.2f}°")
            
            self.ideal_elbow_label.setText(f"{ideal_elbow_deg:.2f}°")
            self.real_elbow_label.setText(f"{real_elbow_deg:.0f}°")
            self.diff_elbow_label.setText(f"{elbow_diff:.2f}°")
            
            self.ideal_wrist_label.setText(f"{ideal_wrist_deg:.2f}°")
            self.real_wrist_label.setText(f"{real_wrist_deg:.0f}°")
            self.diff_wrist_label.setText(f"{wrist_diff:.2f}°")
            
            # Update error labels
            self.ideal_error_label.setText(f"{ideal_error:.4f} cm")
            self.real_error_label.setText(f"{real_error:.4f} cm")
            self.diff_error_label.setText(f"{error_diff:.4f} cm")
            
            # Highlight differences that are significant
            if error_diff > 1.0:
                self.diff_error_label.setStyleSheet("color: red; font-weight: bold; " + self.angle_value_font_style) # Changed here
            else:
                self.diff_error_label.setStyleSheet(self.angle_value_font_style) # Changed here
        # ...
        else:
            # Clear the display if no solutions
            self.ideal_base_label.setText("0.00°")
            self.real_base_label.setText("0°")
            self.diff_base_label.setText("0.00°")
            
            self.ideal_shoulder_label.setText("0.00°")
            self.real_shoulder_label.setText("0°")
            self.diff_shoulder_label.setText("0.00°")
            
            self.ideal_elbow_label.setText("0.00°")
            self.real_elbow_label.setText("0°")
            self.diff_elbow_label.setText("0.00°")
            
            self.ideal_wrist_label.setText("0.00°")
            self.real_wrist_label.setText("0°")
            self.diff_wrist_label.setText("0.00°")
            
            self.ideal_error_label.setText("0.00 cm")
            self.real_error_label.setText("0.00 cm")
            self.diff_error_label.setText("0.00 cm")
            
            self.target_pos_label.setText("(0.00, 0.00, 0.00) cm")

def main():
    app = QApplication(sys.argv)
    window = ComparisonSimulationWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
