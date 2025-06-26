import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.animation as animation
from math import sin, cos, atan2, sqrt, pi
import sys
import os
import subprocess
import threading
import time
import statistics
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QSlider, QLabel, QPushButton, 
                            QLineEdit, QGroupBox, QGridLayout, QTabWidget,
                            QComboBox, QCheckBox, QRadioButton, QButtonGroup,
                            QProgressDialog, QMessageBox,QTextEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, pyqtSlot, QThread, QBuffer, QIODevice # Make sure pyqtSlot is imported
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QImage, QPixmap
import cv2
# Import from the improved robot arm file and path planner
from robot_arm_improved import RobotArm
from path_planner import PathPlanner

# Gemini API 
import google.generativeai as genai




import tkinter as tk
from tkinter import ttk  # For themed widgets like Notebook and Scale
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory # Recommended for smoother servo control
from time import sleep
import sys # To handle exit gracefully
import requests
import json
import threading


RASPBERRY_PI_IP = "192.168.145.251"  # <<< IMPORTANT: REPLACE THIS
RASPBERRY_PI_PORT = 5000 # Default Flask port
PI_URL = f"http://{RASPBERRY_PI_IP}:{RASPBERRY_PI_PORT}/set_angles"
PI_CHASSIS_URL = f"http://{RASPBERRY_PI_IP}:{RASPBERRY_PI_PORT}/move_chassis"
PI_ARM_URL = f"http://{RASPBERRY_PI_IP}:{RASPBERRY_PI_PORT}/set_angles"
PI_START_DETECTION_URL = f"http://{RASPBERRY_PI_IP}:{RASPBERRY_PI_PORT}/start_cube_detection"
PI_STOP_DETECTION_URL = f"http://{RASPBERRY_PI_IP}:{RASPBERRY_PI_PORT}/stop_cube_detection"
PI_GET_CUBE_COORDS_URL = f"http://{RASPBERRY_PI_IP}:{RASPBERRY_PI_PORT}/get_cube_coordinates"
VIDEO_STREAM_URL = f"http://{RASPBERRY_PI_IP}:5000/video_feed"


class ImprovedRobotArmGUI(QMainWindow):
    
    
    # --- Servo Configuration ---
    # Use pigpio factory for potentially smoother control, especially with multiple servos
    # Make sure the pigpiod daemon is running! (sudo systemctl start pigpiod)
    """
    try:
        factory = PiGPIOFactory()
        # Define servos using BCM pin numbers from your image
        # Adjust min_pulse_width and max_pulse_width if necessary for your specific servos
        arm_1 = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
        arm_2 = AngularServo(19, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
        gripper = AngularServo(12, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)
        gripper_wrist = AngularServo(13, min_pulse_width=0.0006, max_pulse_width=0.0023, pin_factory=factory)

        # Store servos in a dictionary for easier access if needed
        servos = {
            "arm_1": arm_1,
            "arm_2": arm_2,
            "gripper": gripper,
            "gripper_wrist": gripper_wrist,
        }
        print("Servos initialized successfully.")

    except Exception as e:
        print(f"Error initializing servos: {e}")
        print("Please ensure the pigpiod daemon is running ('sudo systemctl start pigpiod')")
        print("and the GPIO pins are connected correctly.")
        # Exit if servos can't be initialized
        # You might want to allow the GUI to run in a 'demo' mode without hardware
        # sys.exit(1) # Uncomment this line to force exit on error

    """
    
    # --- Define Signals ---
    # Signal to update the status label (takes string)
    update_status_signal = pyqtSignal(str)
    # Signal to update the progress dialog (takes int)
    update_progress_signal = pyqtSignal(int)
    # Signal when detection is complete (takes avg x, y, z floats)
    detection_complete_signal = pyqtSignal(float, float, float)
    # Signal when detection fails or finishes incompletely (takes string message)
    detection_failed_signal = pyqtSignal(str)
    # Signal to re-enable the button (takes bool) - Can often connect directly
    # enable_button_signal = pyqtSignal(bool) # Not strictly needed if handled in completion/failure slots
    path_execution_status_signal = pyqtSignal(str)
    path_execution_progress_signal = pyqtSignal(int, int) # current_step, total_steps
    path_execution_finished_signal = pyqtSignal(str) # message for completion/cancellation

    
    update_chassis_status_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        # Keep a reference to the progress dialog if needed in slots
        self.progress_dialog = None
        # Flag for cancellation (optional but good practice)
        self.cancel_requested = False
        
        # <<< ADD this line to store the URL >>>
        self.pi_url = f"http://{RASPBERRY_PI_IP}:{RASPBERRY_PI_PORT}/set_angles"
        
        # In __init__ method of ImprovedRobotArmGUI
        self.detection_poll_timer = QTimer()
        self.detection_poll_timer.timeout.connect(self.poll_cube_coordinates_from_pi)
        # How many consecutive valid detections to average over before accepting.
        self.detection_data_points_target = 5 # e.g., average over 5 good readings
        self.collected_detection_points = []
        self.detection_max_poll_attempts = 60 # e.g., try for 30 seconds if polling every 500ms
        self.current_poll_attempts = 0
        # For real arm path execution
        self._path_execution_thread = None
        self._path_execution_worker = None
        self.cancel_path_execution_flag = False        
        
        self.video_thread = None
        self.gemini_thread = None
        self.current_video_qimage = None 
        
        # Create robot arm instance
        self.robot = RobotArm()
        
        # Setup the main window
        self.setWindowTitle("4DoF Robot Arm Simulator (Improved)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create tabs for forward and inverse kinematics
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)
        
        # Create forward kinematics tab
        self.fk_tab = QWidget()
        self.tabs.addTab(self.fk_tab, "Forward Kinematics")
        
        # Create inverse kinematics tab
        self.ik_tab = QWidget()
        self.tabs.addTab(self.ik_tab, "Inverse Kinematics")
        
        # Create multiple solutions tab
        self.solutions_tab = QWidget()
        self.tabs.addTab(self.solutions_tab, "Multiple Solutions")
        
        # --- Create Chassis Control Tab ---
        self.chassis_tab = QWidget()
        self.tabs.addTab(self.chassis_tab, "Chassis Control")
        
        self.live_video_tab = QWidget()
        self.tabs.addTab(self.live_video_tab, "Live Stream and Vision")
        
        # Setup the forward kinematics tab
        self.setup_forward_kinematics_tab()
        
        # Setup the inverse kinematics tab
        self.setup_inverse_kinematics_tab()
        
        # Setup the multiple solutions tab
        self.setup_multiple_solutions_tab()
        
        self.setup_chassis_control_tab()
        
        self.setup_live_video_tab()
        
        # Setup the 3D plot
        self.setup_3d_plot(main_layout)
        
        # Update the plot
        self.update_plot()
        
        # List to store multiple solutions
        self.solutions = []
        self.current_solution_index = 0
        
        # Flag to control animation
        self.animation_running = False
        self.animation_path = []
        self.animation_solutions = []  # Pre-calculated solutions for each point in the path
        self.animation_index = 0
        self.animation_config = None  # Configuration to use for animation (elbow_up, elbow_down, or None for auto)
        
        # Start animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # Update every 100ms
        
        # Connect signals to slots
        self.setup_signals()
        
    
    def update_plot_limits(self):
        if hasattr(self, 'robot') and self.robot:
            max_reach = self.robot.base_height + self.robot.shoulder_length + self.robot.elbow_length + self.robot.gripper_length
            self.ax.set_xlim(-max_reach, max_reach)
            self.ax.set_ylim(-max_reach, max_reach)
            self.ax.set_zlim(0, max_reach)
            self.ax.set_box_aspect([1, 1, 1]) # Eşit en-boy oranı


    def setup_live_video_tab(self):
        tab_layout = QVBoxLayout(self.live_video_tab)

        # Video görüntüleme alanı
        self.live_video_label = QLabel("Press 'Start' for Live Stream.")
        self.live_video_label.setFixedSize(640, 480) # Veya dinamik boyut
        self.live_video_label.setAlignment(Qt.AlignCenter)
        self.live_video_label.setStyleSheet("border: 1px solid black; background-color: #ddd;")
        tab_layout.addWidget(self.live_video_label)

        # Video durumu etiketi
        self.live_video_status_label = QLabel("Status: Ready")
        tab_layout.addWidget(self.live_video_status_label)

        # Video başlat/durdur butonu
        self.toggle_live_video_button = QPushButton("Start Video Stream")
        self.toggle_live_video_button.setCheckable(True)
        self.toggle_live_video_button.toggled.connect(self.handle_toggle_video_stream)
        tab_layout.addWidget(self.toggle_live_video_button)

        # Gemini'ye Sorma Butonu
        self.ask_gemini_button = QPushButton("What is in this frame?")
        self.ask_gemini_button.clicked.connect(self.handle_ask_gemini)
        self.ask_gemini_button.setEnabled(False) # Video başlayana kadar pasif
        tab_layout.addWidget(self.ask_gemini_button)

        # Gemini cevap alanı (kaydırılabilir olması için QTextEdit)
        self.gemini_response_text = QTextEdit()
        self.gemini_response_text.setReadOnly(True)
        self.gemini_response_text.setPlaceholderText("Vision response will appear here.")
        self.gemini_response_text.setFixedHeight(150) # Yüksekliği ayarla
        tab_layout.addWidget(self.gemini_response_text)

        tab_layout.addStretch() # Kalan alanı ittir

    @pyqtSlot(bool)
    def handle_toggle_video_stream(self, checked):
        if checked: # Başlat
            if self.video_thread is None or not self.video_thread.isRunning():
                self.live_video_label.setText("Connecting...")
                self.live_video_status_label.setText("Status: Connecting...")
                self.video_thread = VideoThread(VIDEO_STREAM_URL)
                self.video_thread.change_pixmap_signal.connect(self.update_live_video_frame)
                self.video_thread.connection_status_signal.connect(self.update_live_video_status)
                self.video_thread.finished.connect(self.video_stream_finished)
                self.video_thread.start()
                self.toggle_live_video_button.setText("Stop Video Stream")
                self.ask_gemini_button.setEnabled(False) # Henüz kare yok
        else: # Durdur
            if self.video_thread and self.video_thread.isRunning():
                self.live_video_status_label.setText("Status: Stopping...")
                self.video_thread.stop()
                # video_stream_finished slotu buton metnini ve durumu güncelleyecek.
                self.ask_gemini_button.setEnabled(False)


    @pyqtSlot(QImage)
    def update_live_video_frame(self, q_image):
        self.current_video_qimage = q_image.copy() # Gemini için kareyi sakla
        # Görüntüyü QLabel'in boyutuna sığdır ama en-boy oranını koru
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.live_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.live_video_label.setPixmap(scaled_pixmap)
        if not self.ask_gemini_button.isEnabled() and self.toggle_live_video_button.isChecked():
             self.ask_gemini_button.setEnabled(True) # İlk kare geldi, butonu aktif et

    @pyqtSlot(str)
    def update_live_video_status(self, status):
        self.live_video_status_label.setText(f"Durum: {status}")
        if "Hata" in status or "Durduruldu" in status:
            if self.toggle_live_video_button.isChecked(): # Eğer kullanıcı başlatmışsa ama hata oluştuysa
                self.toggle_live_video_button.setChecked(False) # Buton durumunu düzelt
            self.ask_gemini_button.setEnabled(False)
            if "Hata" in status:
                self.live_video_label.setText("Video Yüklenemedi / Hata")

    def video_stream_finished(self):
        self.live_video_status_label.setText("Status: Stream finished (Thread is over)")
        self.toggle_live_video_button.setText("Start Video Stream")
        if self.toggle_live_video_button.isChecked(): # Kullanıcı durdurmadıysa
             self.toggle_live_video_button.setChecked(False)
        self.current_video_qimage = None
        self.ask_gemini_button.setEnabled(False)
        self.live_video_label.setText("Press 'Start' for Live Stream..") # Placeholder

    def handle_ask_gemini(self):
        if self.current_video_qimage is None or self.current_video_qimage.isNull():
            QMessageBox.warning(self, "Hata", "Gemini'ye göndermek için geçerli bir video karesi yok.")
            return

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            QMessageBox.critical(self, "API Anahtarı Eksik",
                                 "GEMINI_API_KEY ortam değişkeni ayarlanmamış.\n"
                                 "Lütfen API anahtarınızı ayarlayın ve uygulamayı yeniden başlatın.")
            return

        if not genai:
            QMessageBox.critical(self, "Kütüphane Eksik",
                                 "Google Generative AI kütüphanesi bulunamadı. Lütfen yükleyin.")
            return

        # QImage'i JPEG byte'larına çevir
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        # Kaliteyi düşürmek dosya boyutunu azaltır, API'ye gönderme hızını artırabilir
        self.current_video_qimage.save(buffer, "JPEG", 80) # %80 kalite
        image_bytes = buffer.data().data() # QByteArray'den bytes
        buffer.close()

        if not image_bytes:
            QMessageBox.warning(self, "Hata", "Video karesi JPEG formatına dönüştürülemedi.")
            return

        prompt = "What do you see in this image? Describe it in as much detail as possible."

        self.gemini_response_text.setPlaceholderText("Red Rover is answering, Please wait...")
        self.gemini_response_text.clear() # Önceki cevabı temizle
        self.ask_gemini_button.setEnabled(False) # Tekrar tıklamayı engelle

        self.gemini_thread = GeminiVisionThread(api_key, image_bytes, prompt)
        self.gemini_thread.gemini_response_signal.connect(self.display_gemini_response)
        self.gemini_thread.gemini_error_signal.connect(self.display_gemini_error)
        self.gemini_thread.finished.connect(self.gemini_task_finished) # Thread bitince
        self.gemini_thread.start()

    @pyqtSlot(str)
    def display_gemini_response(self, response_text):
        self.gemini_response_text.setText(response_text)
        # self.ask_gemini_button.setEnabled(True) # gemini_task_finished içinde yapılacak

    @pyqtSlot(str)
    def display_gemini_error(self, error_text):
        self.gemini_response_text.setPlaceholderText("Gemini'den cevap alınamadı.")
        self.gemini_response_text.setText(f"Bir Hata Oluştu:\n{error_text}")
        # self.ask_gemini_button.setEnabled(True) # gemini_task_finished içinde yapılacak

    def gemini_task_finished(self):
        # Gemini butonu, video akışı hala aktifse ve yeni bir kare varsa tekrar aktif olmalı.
        if self.video_thread and self.video_thread.isRunning() and self.current_video_qimage:
            self.ask_gemini_button.setEnabled(True)
        else:
            self.ask_gemini_button.setEnabled(False)
        print("Gemini thread sonlandı.")


    def closeEvent(self, event):
        print("Uygulama kapatılıyor...")
        if self.video_thread and self.video_thread.isRunning():
            print("Video thread durduruluyor...")
            self.video_thread.stop() # Bu wait() çağırır.
        if self.gemini_thread and self.gemini_thread.isRunning():
            print("Gemini thread bekleniyor (genellikle hızlı biter)...")
            self.gemini_thread.wait(1000) # Çok uzun sürmemeli

        super().closeEvent(event)
    
    
    
    def send_angles_to_pi(self, angles_dict):
        """Sends the target angles to the Raspberry Pi server in a separate thread."""
        
        def _send_request():
            """Worker function to run in a separate thread."""
            try:
                # Add a timeout to prevent the GUI from freezing indefinitely
                response = requests.post(self.pi_url, json=angles_dict, timeout=0.5) # Short timeout (0.5 seconds)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                print(f"Sent to Pi: {angles_dict}, Response: {response.json()}")
                # Optionally update a status label on success (using signals if needed)
                # self.update_status_signal.emit("Sent angles to Pi successfully.")
            except requests.exceptions.Timeout:
                print(f"Error sending angles to Pi: Connection timed out (is {self.pi_url} correct and server running?)")
                # Optionally update a status label on failure (using signals if needed)
                # self.update_status_signal.emit("Error: Pi connection timed out.")
            except requests.exceptions.RequestException as e:
                print(f"Error sending angles to Pi: {e}")
                # Optionally update a status label on failure (using signals if needed)
                # self.update_status_signal.emit(f"Error: Pi connection failed ({e})")
            except Exception as e:
                # Catch any other unexpected errors
                print(f"An unexpected error occurred during Pi communication: {e}")
                # self.update_status_signal.emit("Error: Unexpected communication error.")

        # Run the network request in a separate thread to avoid blocking the GUI
        # Especially important for the FK sliders!
        thread = threading.Thread(target=_send_request, daemon=True)
        thread.start()
        
    # --- NEW: Function to send Chassis Commands ---
    def send_chassis_command(self, command, speed):
        """Sends a movement command and speed for the chassis to the Raspberry Pi."""

        # Validate speed
        try:
            speed_float = float(speed)
            if not (0.0 <= speed_float <= 1.0):
                raise ValueError("Speed must be between 0.0 and 1.0")
        except ValueError as e:
            print(f"Invalid speed value: {speed}. Error: {e}")
            self.update_chassis_status_signal.emit(f"Error: Invalid Speed ({speed})")
            return # Don't send invalid command

        command_data = {"command": command, "speed": speed_float}

        def _send_request():
            """Worker function for chassis commands."""
            try:
                response = requests.post(PI_CHASSIS_URL, json=command_data, timeout=1.0) # Longer timeout maybe needed?
                response.raise_for_status()
                print(f"Sent Chassis Command: {command_data}, Response: {response.json()}")
                self.update_chassis_status_signal.emit(f"Command Sent: {command} @ {speed_float*100:.0f}%")
            except requests.exceptions.Timeout:
                error_msg = f"Error: Chassis connection timed out (Check {PI_CHASSIS_URL})"
                print(error_msg)
                self.update_chassis_status_signal.emit(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"Error: Chassis connection failed ({e})"
                print(error_msg)
                self.update_chassis_status_signal.emit(error_msg)
            except Exception as e:
                error_msg = f"Unexpected chassis communication error: {e}"
                print(error_msg)
                self.update_chassis_status_signal.emit(error_msg)

        thread = threading.Thread(target=_send_request, daemon=True)
        thread.start()
        
        
    def setup_forward_kinematics_tab(self):
        """Setup the forward kinematics tab with sliders for joint control"""
        layout = QVBoxLayout(self.fk_tab)
        
        # Create sliders for joint angles
        slider_group = QGroupBox("Joint Control")
        slider_layout = QGridLayout()
        slider_group.setLayout(slider_layout)
        
        # Base rotation slider (around z-axis)
        slider_layout.addWidget(QLabel("Base Rotation:"), 0, 0)
        self.base_slider = QSlider(Qt.Horizontal)
        self.base_slider.setRange(-90, 90)
        self.base_slider.setValue(0)
        self.base_slider.setTickPosition(QSlider.TicksBelow)
        self.base_slider.setTickInterval(30)
        self.base_slider.valueChanged.connect(self.update_joint_angles)
        slider_layout.addWidget(self.base_slider, 0, 1)
        self.base_value_label = QLabel("0°")
        slider_layout.addWidget(self.base_value_label, 0, 2)
        
        # Shoulder rotation slider (around y-axis)
        slider_layout.addWidget(QLabel("Shoulder Rotation:"), 1, 0)
        self.shoulder_slider = QSlider(Qt.Horizontal)
        self.shoulder_slider.setRange(-90, 90)
        self.shoulder_slider.setValue(0)
        self.shoulder_slider.setTickPosition(QSlider.TicksBelow)
        self.shoulder_slider.setTickInterval(15)
        self.shoulder_slider.valueChanged.connect(self.update_joint_angles)
        slider_layout.addWidget(self.shoulder_slider, 1, 1)
        self.shoulder_value_label = QLabel("0°")
        slider_layout.addWidget(self.shoulder_value_label, 1, 2)
        
        # Elbow rotation slider (around y-axis)
        slider_layout.addWidget(QLabel("Elbow Rotation:"), 2, 0)
        self.elbow_slider = QSlider(Qt.Horizontal)
        self.elbow_slider.setRange(-90, 90)
        self.elbow_slider.setValue(0)
        self.elbow_slider.setTickPosition(QSlider.TicksBelow)
        self.elbow_slider.setTickInterval(15)
        self.elbow_slider.valueChanged.connect(self.update_joint_angles)
        slider_layout.addWidget(self.elbow_slider, 2, 1)
        self.elbow_value_label = QLabel("0°")
        slider_layout.addWidget(self.elbow_value_label, 2, 2)
        
        # Wrist rotation slider (around y-axis)
        slider_layout.addWidget(QLabel("Wrist Rotation:"), 3, 0)
        self.wrist_slider = QSlider(Qt.Horizontal)
        self.wrist_slider.setRange(-90, 90)
        self.wrist_slider.setValue(0)
        self.wrist_slider.setTickPosition(QSlider.TicksBelow)
        self.wrist_slider.setTickInterval(15)
        self.wrist_slider.valueChanged.connect(self.update_joint_angles)
        slider_layout.addWidget(self.wrist_slider, 3, 1)
        self.wrist_value_label = QLabel("0°")
        slider_layout.addWidget(self.wrist_value_label, 3, 2)
        
        # Gripper state slider
        slider_layout.addWidget(QLabel("Gripper (Open/Close):"), 4, 0)
        self.gripper_slider = QSlider(Qt.Horizontal)
        self.gripper_slider.setRange(0, 100)
        self.gripper_slider.setValue(0)
        self.gripper_slider.setTickPosition(QSlider.TicksBelow)
        self.gripper_slider.setTickInterval(10)
        self.gripper_slider.valueChanged.connect(self.update_joint_angles)
        slider_layout.addWidget(self.gripper_slider, 4, 1)
        self.gripper_value_label = QLabel("Closed")
        slider_layout.addWidget(self.gripper_value_label, 4, 2)
        
        layout.addWidget(slider_group)
        
        # End effector position display
        position_group = QGroupBox("End Effector Position")
        position_layout = QGridLayout()
        position_group.setLayout(position_layout)
        
        position_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_position_label = QLabel("0.00 cm")
        position_layout.addWidget(self.x_position_label, 0, 1)
        
        position_layout.addWidget(QLabel("Y:"), 1, 0)
        self.y_position_label = QLabel("0.00 cm")
        position_layout.addWidget(self.y_position_label, 1, 1)
        
        position_layout.addWidget(QLabel("Z:"), 2, 0)
        self.z_position_label = QLabel("0.00 cm")
        position_layout.addWidget(self.z_position_label, 2, 1)
        
        layout.addWidget(position_group)
        

        
        # End effector movement controls for forward kinematics
        fk_movement_group = QGroupBox("End Effector Movement Controls")
        fk_movement_layout = QGridLayout()
        fk_movement_group.setLayout(fk_movement_layout)
        
        # Up button
        self.fk_up_button = QPushButton("Up (Z+)")
        self.fk_up_button.clicked.connect(self.move_end_effector_up)
        fk_movement_layout.addWidget(self.fk_up_button, 0, 1)
        
        # Down button
        self.fk_down_button = QPushButton("Down (Z-)")
        self.fk_down_button.clicked.connect(self.move_end_effector_down)
        fk_movement_layout.addWidget(self.fk_down_button, 2, 1)
        
        # Left button
        self.fk_left_button = QPushButton("Left (Y+)")
        self.fk_left_button.clicked.connect(self.move_end_effector_left)
        fk_movement_layout.addWidget(self.fk_left_button, 1, 0)
        
        # Right button
        self.fk_right_button = QPushButton("Right (Y-)")
        self.fk_right_button.clicked.connect(self.move_end_effector_right)
        fk_movement_layout.addWidget(self.fk_right_button, 1, 2)
        
        # Forward button
        self.fk_forward_button = QPushButton("Forward (X+)")
        self.fk_forward_button.clicked.connect(self.move_end_effector_forward)
        fk_movement_layout.addWidget(self.fk_forward_button, 1, 3)
        
        # Backward button
        self.fk_backward_button = QPushButton("Backward (X-)")
        self.fk_backward_button.clicked.connect(self.move_end_effector_backward)
        fk_movement_layout.addWidget(self.fk_backward_button, 1, 4)
        
        # Movement step size control
        fk_movement_layout.addWidget(QLabel("Step Size:"), 3, 0)
        self.fk_step_size_input = QLineEdit("0.5")
        self.fk_step_size_input.setMaximumWidth(60)
        self.fk_step_size_input.textChanged.connect(self.sync_step_size_inputs)
        fk_movement_layout.addWidget(self.fk_step_size_input, 3, 1)
        fk_movement_layout.addWidget(QLabel("cm"), 3, 2)
        
        # Add movement controls to layout
        layout.addWidget(fk_movement_group)
        
        # Calibration offsets
        offset_group = QGroupBox("Calibration Offsets")
        offset_layout = QGridLayout()
        offset_group.setLayout(offset_layout)
        
        # Base offset
        offset_layout.addWidget(QLabel("Base Offset:"), 0, 0)
        self.base_offset_input = QLineEdit("0")
        self.base_offset_input.setMaximumWidth(50)
        offset_layout.addWidget(self.base_offset_input, 0, 1)
        offset_layout.addWidget(QLabel("°"), 0, 2)
        
        # Shoulder offset
        offset_layout.addWidget(QLabel("Shoulder Offset:"), 1, 0)
        self.shoulder_offset_input = QLineEdit("0")
        self.shoulder_offset_input.setMaximumWidth(50)
        offset_layout.addWidget(self.shoulder_offset_input, 1, 1)
        offset_layout.addWidget(QLabel("°"), 1, 2)
        
        # Elbow offset
        offset_layout.addWidget(QLabel("Elbow Offset:"), 2, 0)
        self.elbow_offset_input = QLineEdit("0")
        self.elbow_offset_input.setMaximumWidth(50)
        offset_layout.addWidget(self.elbow_offset_input, 2, 1)
        offset_layout.addWidget(QLabel("°"), 2, 2)
        
        # Wrist offset
        offset_layout.addWidget(QLabel("Wrist Offset:"), 3, 0)
        self.wrist_offset_input = QLineEdit("0")
        self.wrist_offset_input.setMaximumWidth(50)
        offset_layout.addWidget(self.wrist_offset_input, 3, 1)
        offset_layout.addWidget(QLabel("°"), 3, 2)
        
        # Apply offsets button
        apply_offset_button = QPushButton("Apply Offsets")
        apply_offset_button.clicked.connect(self.apply_offsets)
        offset_layout.addWidget(apply_offset_button, 4, 0, 1, 3)
        
        layout.addWidget(offset_group)
        
        # Image detection functionality
        image_detection_group = QGroupBox("Image Detection")
        image_detection_layout = QGridLayout()
        image_detection_group.setLayout(image_detection_layout)
        
        # Image detection button
        self.image_detection_button = QPushButton("Run Image Detection")
        self.image_detection_button.clicked.connect(self.run_image_detection)
        image_detection_layout.addWidget(self.image_detection_button, 0, 0, 1, 2)
        
        # Status label for image detection
        self.image_detection_status = QLabel("Status: Ready")
        image_detection_layout.addWidget(self.image_detection_status, 1, 0, 1, 2)
        
        layout.addWidget(image_detection_group)
        
        # Add spacer to push everything to the top
        layout.addStretch()
    
    def run_image_detection(self):
        """Run the image detection process using the Raspberry Pi's camera."""
        self.image_detection_button.setEnabled(False)
        self.cancel_requested = False
        self.collected_detection_points = []
        self.current_poll_attempts = 0
        self.update_status_signal.emit("Status: Requesting detection start on Pi...")

        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Create progress dialog
        self.progress_dialog = QProgressDialog("Starting detection on Pi...", "Cancel", 0, self.detection_data_points_target, self)
        self.progress_dialog.setWindowTitle("Pi Image Detection")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

        # Start detection on Pi
        try:
            response = requests.post(PI_START_DETECTION_URL, timeout=10)
            response.raise_for_status()
            json_response = response.json()
            
            if json_response.get("status") == "success":
                self.update_status_signal.emit("Status: Pi detection started. Polling for cube...")
                # Start polling timer
                QTimer.singleShot(100, lambda: self.detection_poll_timer.start(500))
            else:
                self.detection_failed_signal.emit(f"Pi Error: {json_response.get('message', 'Failed to start')}")
                self.image_detection_button.setEnabled(True)
        except requests.exceptions.RequestException as e:
            self.detection_failed_signal.emit(f"Error starting Pi detection: {str(e)}")
            self.image_detection_button.setEnabled(True)
        except Exception as e:
            self.detection_failed_signal.emit(f"Unexpected error on start: {str(e)}")
            self.image_detection_button.setEnabled(True)

    def poll_cube_coordinates_from_pi(self):
        """Poll the Raspberry Pi for cube coordinates."""
        if self.cancel_requested:
            self._finalize_pi_detection("Canceled by user.", success=False)
            return

        self.current_poll_attempts += 1
        if self.current_poll_attempts > self.detection_max_poll_attempts and not self.collected_detection_points:
            self._finalize_pi_detection("Timeout: No cube detected by Pi.", success=False)
            return

        try:
            response = requests.get(PI_GET_CUBE_COORDS_URL, timeout=0.4)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success" and data.get("coordinates"):
                coords = data["coordinates"]
                self.collected_detection_points.append(coords)

                # Update progress based on collected points vs target
                progress_val = min(len(self.collected_detection_points), self.detection_data_points_target)
                self.update_progress_signal.emit(progress_val)
                self.update_status_signal.emit(f"Status: Cube found by Pi. Reading {len(self.collected_detection_points)}/{self.detection_data_points_target}...")

                if len(self.collected_detection_points) >= self.detection_data_points_target:
                    avg_x = statistics.mean([p[0] for p in self.collected_detection_points])
                    avg_y = statistics.mean([p[1] for p in self.collected_detection_points])
                    avg_z = statistics.mean([p[2] for p in self.collected_detection_points])

                    print_msg = (f"GUI: AVG COORDS FROM PI: X={avg_x:.2f}, Y={avg_y:.2f}, Z={avg_z:.2f} "
                                f"(from {len(self.collected_detection_points)} readings)")
                    print(print_msg)
                    self._finalize_pi_detection(print_msg, success=True, coords=(avg_x, avg_y, avg_z))
            elif data.get("status") == "not_found":
                self.update_status_signal.emit(f"Status: Polling Pi ({self.current_poll_attempts})... Cube not detected.")
        except requests.exceptions.Timeout:
            self.update_status_signal.emit(f"Status: Pi coord request timeout ({self.current_poll_attempts})...")
        except requests.exceptions.RequestException as e:
            self._finalize_pi_detection(f"Polling error: {str(e)[:100]}", success=False)
        except Exception as e:
            self._finalize_pi_detection(f"Unexpected polling error: {str(e)[:100]}", success=False)

    def _finalize_pi_detection(self, message, success, coords=None):
        """Finalize the Pi detection process."""
        self.detection_poll_timer.stop()

        # Try to tell Pi to stop its detection loop
        try:
            requests.post(PI_STOP_DETECTION_URL, timeout=3)
            print("PC: Sent stop detection command to Pi.")
        except requests.exceptions.RequestException as e:
            print(f"PC: Note - Could not send stop command to Pi: {e}")

        if success and coords:
            self.detection_complete_signal.emit(coords[0], coords[1], coords[2])
        else:
            self.detection_failed_signal.emit(message)

    def cancel_pi_detection(self):
        """Cancel the Pi detection process."""
        print("PC GUI: Pi Detection Canceled by User.")
        self.cancel_requested = True
        if not self.detection_poll_timer.isActive():
            self._finalize_pi_detection("Canceled by user before polling started.", success=False)
    
    # --- NEW: Setup Chassis Control Tab ---
    def setup_chassis_control_tab(self):
        """Setup the chassis control tab with movement buttons and speed slider"""
        layout = QVBoxLayout(self.chassis_tab)

        # Speed Control Group
        speed_group = QGroupBox("Speed Control")
        speed_layout = QHBoxLayout()
        speed_group.setLayout(speed_layout)

        speed_layout.addWidget(QLabel("Speed:"))
        self.chassis_speed_slider = QSlider(Qt.Horizontal)
        self.chassis_speed_slider.setRange(0, 100) # 0% to 100%
        self.chassis_speed_slider.setValue(50) # Default 50%
        self.chassis_speed_slider.setTickPosition(QSlider.TicksBelow)
        self.chassis_speed_slider.setTickInterval(10)
        speed_layout.addWidget(self.chassis_speed_slider)

        self.chassis_speed_label = QLabel("50%")
        self.chassis_speed_slider.valueChanged.connect(
            lambda value: self.chassis_speed_label.setText(f"{value}%")
        )
        speed_layout.addWidget(self.chassis_speed_label)
        layout.addWidget(speed_group)

        # Movement Control Group
        movement_group = QGroupBox("Movement")
        movement_layout = QGridLayout()
        movement_group.setLayout(movement_layout)

        # --- Create Buttons ---
        self.forward_button = QPushButton("Forward")
        self.backward_button = QPushButton("Backward")
        self.left_button = QPushButton("Left")
        self.right_button = QPushButton("Right")
        self.turn_left_button = QPushButton("Turn Left")
        self.turn_right_button = QPushButton("Turn Right")
        self.stop_button = QPushButton("STOP")
        self.stop_button.setStyleSheet("background-color: red; color: white;") # Make stop obvious

        # --- Arrange Buttons ---
        # Row 0: Turn Left, Forward, Turn Right
        movement_layout.addWidget(self.turn_left_button, 0, 0)
        movement_layout.addWidget(self.forward_button, 0, 1)
        movement_layout.addWidget(self.turn_right_button, 0, 2)
        # Row 1: Left, STOP, Right
        movement_layout.addWidget(self.left_button, 1, 0)
        movement_layout.addWidget(self.stop_button, 1, 1)
        movement_layout.addWidget(self.right_button, 1, 2)
        # Row 2: (Empty), Backward, (Empty)
        movement_layout.addWidget(self.backward_button, 2, 1)

        layout.addWidget(movement_group)

        # --- Chassis Status Label ---
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        self.chassis_status_label = QLabel("Status: Ready")
        status_layout.addWidget(self.chassis_status_label)
        layout.addWidget(status_group)


        # --- Connect Buttons ---
        # Use lambda to pass command string and get speed from slider
        self.forward_button.clicked.connect(lambda: self.send_chassis_command("forward", self.get_chassis_speed()))
        self.backward_button.clicked.connect(lambda: self.send_chassis_command("backward", self.get_chassis_speed()))
        self.left_button.clicked.connect(lambda: self.send_chassis_command("left", self.get_chassis_speed()))
        self.right_button.clicked.connect(lambda: self.send_chassis_command("right", self.get_chassis_speed()))
        self.turn_left_button.clicked.connect(lambda: self.send_chassis_command("turn_left", self.get_chassis_speed()))
        self.turn_right_button.clicked.connect(lambda: self.send_chassis_command("turn_right", self.get_chassis_speed()))
        # Stop command usually doesn't need speed, but we send 0 for consistency
        self.stop_button.clicked.connect(lambda: self.send_chassis_command("forward", 0.0))

        # Add spacer
        layout.addStretch()

    # --- NEW: Helper to get chassis speed ---
    def get_chassis_speed(self):
        """Gets the current speed from the chassis slider (0.0 to 1.0)."""
        slider_value = self.chassis_speed_slider.value()
        # Normalleştirilmiş değer: 0.0 - 1.0
        normalized_speed = slider_value / 100.0
        
        return normalized_speed
    
    def setup_signals(self):
        """Connect signals to their corresponding slots."""
        self.update_status_signal.connect(self.update_status_slot)
        # You might connect update_progress_signal directly to the dialog if it's safe,
        # but using a slot is safer if you need other logic.
        self.update_progress_signal.connect(self.update_progress_slot)
        self.detection_complete_signal.connect(self.detection_complete_slot)
        self.detection_failed_signal.connect(self.detection_failed_slot)
        # self.enable_button_signal.connect(self.image_detection_button.setEnabled)
        self.update_chassis_status_signal.connect(self.update_chassis_status_slot)
        
        self.path_execution_status_signal.connect(self.update_path_execution_status_slot)
        self.path_execution_progress_signal.connect(self.update_path_execution_progress_slot)
        self.path_execution_finished_signal.connect(self.path_execution_finished_slot)

    # --- Define Slots ---
    @pyqtSlot(str)
    def update_status_slot(self, text):
        """Safely updates the status label from the main thread."""
        self.image_detection_status.setText(text)

    @pyqtSlot(int)
    def update_progress_slot(self, value):
        """Safely updates the progress dialog from the main thread."""
        if self.progress_dialog:
            self.progress_dialog.setValue(value)

    @pyqtSlot(float, float, float)
    def detection_complete_slot(self, avg_x, avg_y, avg_z):
        """Handles successful detection results in the main thread."""
        self.image_detection_status.setText(f"Status: Pi Cube at X={avg_x:.2f}, Y={avg_y:.2f}, Z={avg_z:.2f} cm")
        if self.progress_dialog:
            self.progress_dialog.setValue(self.detection_data_points_target) # Mark as complete
            self.progress_dialog.close() # Close it
            self.progress_dialog = None

        # Update the target position inputs in the IK tab
        self.x_target_input.setText(f"{avg_x:.2f}")
        self.y_target_input.setText(f"{avg_y:.2f}")
        self.z_target_input.setText(f"{avg_z:.2f}")

        # Switch to IK tab and solve IK without applying
        self.tabs.setCurrentIndex(1)  # Switch to IK tab
        self.solve_inverse_kinematics()  # Solve IK for the detected position

        self.image_detection_status.setText(f"Status: Target position updated. Ready to move arm.")
        self.image_detection_button.setEnabled(True)
        self.collected_detection_points = [] # Clear for next run

    @pyqtSlot(str)
    def detection_failed_slot(self, message):
        """Handles detection failure or incomplete results in the main thread."""
        self.image_detection_status.setText(f"Status: {message}")
        if self.progress_dialog:
             self.progress_dialog.setValue(self.detection_data_points_target) # Mark as complete
             self.progress_dialog.close() # Close it
        self.image_detection_button.setEnabled(True)
        self.collected_detection_points = [] # Clear
        # Ensure polling stops if it hasn't already
        if self.detection_poll_timer.isActive():
            self.detection_poll_timer.stop()
        
    # --- NEW: Slot for Chassis Status ---
    @pyqtSlot(str)
    def update_chassis_status_slot(self, text):
        """Safely updates the chassis status label from the main thread."""
        if hasattr(self, 'chassis_status_label'): # Check if label exists
            self.chassis_status_label.setText(f"Status: {text}")

    @pyqtSlot(str)
    def update_path_execution_status_slot(self, message):
        if hasattr(self, 'path_execution_status_label'): # Check if label exists
            self.path_execution_status_label.setText(f"Real Arm: {message}")

    @pyqtSlot(int, int)
    def update_path_execution_progress_slot(self, current_step, total_steps):
        if self.progress_dialog and self.progress_dialog.isVisible():
            self.progress_dialog.setLabelText(f"Executing path on real arm: Step {current_step}/{total_steps}")
            self.progress_dialog.setValue(current_step)

    @pyqtSlot(str)
    def path_execution_finished_slot(self, message):
        if hasattr(self, 'path_execution_status_label'):
            self.path_execution_status_label.setText(f"Real Arm: {message}")
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        if hasattr(self, 'follow_path_real_button'):
            self.follow_path_real_button.setEnabled(True)
        QMessageBox.information(self, "Real Arm Path Execution", message)

    def cancel_detection(self):
        """Slot connected to the progress dialog's cancel button."""
        print("Cancellation requested.")
        self.cancel_requested = True
        # Note: Stopping the thread cleanly, especially when blocked on
        # process.stdout.readline(), is tricky. The flag is the first step.
        # You might need to terminate the subprocess more forcefully here if needed.
        # self.detection_failed_slot("Detection Canceled by User") # Emit failure signal or handle directly
              
    def setup_inverse_kinematics_tab(self):
        """Setup the improved inverse kinematics tab with target position inputs"""
        layout = QVBoxLayout(self.ik_tab)
        
        # Target position input
        target_group = QGroupBox("Target Position")
        target_layout = QGridLayout()
        target_group.setLayout(target_layout)
        
        # X position
        target_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_target_input = QLineEdit("15")
        self.x_target_input.setMaximumWidth(50)
        target_layout.addWidget(self.x_target_input, 0, 1)
        target_layout.addWidget(QLabel("cm"), 0, 2)
        
        # Y position
        target_layout.addWidget(QLabel("Y:"), 1, 0)
        self.y_target_input = QLineEdit("0")
        self.y_target_input.setMaximumWidth(50)
        target_layout.addWidget(self.y_target_input, 1, 1)
        target_layout.addWidget(QLabel("cm"), 1, 2)
        
        # Z position
        target_layout.addWidget(QLabel("Z:"), 2, 0)
        self.z_target_input = QLineEdit("20")
        self.z_target_input.setMaximumWidth(50)
        target_layout.addWidget(self.z_target_input, 2, 1)
        target_layout.addWidget(QLabel("cm"), 2, 2)
        
        # Configuration options
        target_layout.addWidget(QLabel("Configuration:"), 3, 0)
        self.config_combobox = QComboBox()
        self.config_combobox.addItems(["Elbow Up", "Auto (Best)","Elbow Down"])
        target_layout.addWidget(self.config_combobox, 3, 1, 1, 2)
        
        # Orientation control
        target_layout.addWidget(QLabel("End Effector Orientation:"), 4, 0)
        self.orientation_enabled = QCheckBox("Enable")
        target_layout.addWidget(self.orientation_enabled, 4, 1)
        
        # Orientation angle
        target_layout.addWidget(QLabel("Orientation:"), 5, 0)
        self.orientation_input = QLineEdit("0")
        self.orientation_input.setMaximumWidth(50)
        self.orientation_input.setEnabled(False)
        self.orientation_enabled.stateChanged.connect(
            lambda state: self.orientation_input.setEnabled(state == Qt.Checked)
        )
        target_layout.addWidget(self.orientation_input, 5, 1)
        target_layout.addWidget(QLabel("°"), 5, 2)
        
        # Advanced options
        target_layout.addWidget(QLabel("Error Threshold:"), 6, 0)
        self.error_threshold_input = QLineEdit("0.1")
        self.error_threshold_input.setMaximumWidth(50)
        target_layout.addWidget(self.error_threshold_input, 6, 1)
        target_layout.addWidget(QLabel("cm"), 6, 2)
        
        target_layout.addWidget(QLabel("Max Iterations:"), 7, 0)
        self.max_iterations_input = QLineEdit("100")
        self.max_iterations_input.setMaximumWidth(50)
        target_layout.addWidget(self.max_iterations_input, 7, 1)
        
        # Solve IK button
        solve_button = QPushButton("Solve Inverse Kinematics")
        solve_button.clicked.connect(self.solve_inverse_kinematics)
        target_layout.addWidget(solve_button, 8, 0, 1, 3)
        
        # Find all solutions button
        find_all_button = QPushButton("Find All Solutions")
        find_all_button.clicked.connect(self.find_all_solutions)
        target_layout.addWidget(find_all_button, 9, 0, 1, 3)
        
        layout.addWidget(target_group)
        
        # Solution display
        solution_group = QGroupBox("Joint Angles Solution")
        solution_layout = QGridLayout()
        solution_group.setLayout(solution_layout)
        
        solution_layout.addWidget(QLabel("Base:"), 0, 0)
        self.base_solution_label = QLabel("0.00°")
        solution_layout.addWidget(self.base_solution_label, 0, 1)
        
        solution_layout.addWidget(QLabel("Shoulder:"), 1, 0)
        self.shoulder_solution_label = QLabel("0.00°")
        solution_layout.addWidget(self.shoulder_solution_label, 1, 1)
        
        solution_layout.addWidget(QLabel("Elbow:"), 2, 0)
        self.elbow_solution_label = QLabel("0.00°")
        solution_layout.addWidget(self.elbow_solution_label, 2, 1)
        
        solution_layout.addWidget(QLabel("Wrist:"), 3, 0)
        self.wrist_solution_label = QLabel("0.00°")
        solution_layout.addWidget(self.wrist_solution_label, 3, 1)
        
        solution_layout.addWidget(QLabel("Configuration:"), 4, 0)
        self.config_solution_label = QLabel("None")
        solution_layout.addWidget(self.config_solution_label, 4, 1)
        
        solution_layout.addWidget(QLabel("Error:"), 5, 0)
        self.error_solution_label = QLabel("0.00 cm")
        solution_layout.addWidget(self.error_solution_label, 5, 1)
        
        solution_layout.addWidget(QLabel("Status:"), 6, 0)
        self.ik_status_label = QLabel("Ready")
        solution_layout.addWidget(self.ik_status_label, 6, 1)
        
        # Apply solution button
        apply_solution_button = QPushButton("Apply Solution")
        apply_solution_button.clicked.connect(self.apply_ik_solution)
        solution_layout.addWidget(apply_solution_button, 7, 0, 1, 2)
        
        layout.addWidget(solution_group)
        

        
        # Dimensions adjustment
        dimensions_group = QGroupBox("Arm Dimensions")
        dimensions_layout = QGridLayout()
        dimensions_group.setLayout(dimensions_layout)
        
        # Base height
        dimensions_layout.addWidget(QLabel("Base Height:"), 0, 0)
        self.base_height_input = QLineEdit(str(self.robot.base_height))
        self.base_height_input.setMaximumWidth(50)
        dimensions_layout.addWidget(self.base_height_input, 0, 1)
        dimensions_layout.addWidget(QLabel("cm"), 0, 2)
        
        # Shoulder length
        dimensions_layout.addWidget(QLabel("Shoulder Length:"), 1, 0)
        self.shoulder_length_input = QLineEdit(str(self.robot.shoulder_length))
        self.shoulder_length_input.setMaximumWidth(50)
        dimensions_layout.addWidget(self.shoulder_length_input, 1, 1)
        dimensions_layout.addWidget(QLabel("cm"), 1, 2)
        
        # Elbow length
        dimensions_layout.addWidget(QLabel("Elbow Length:"), 2, 0)
        self.elbow_length_input = QLineEdit(str(self.robot.elbow_length))
        self.elbow_length_input.setMaximumWidth(50)
        dimensions_layout.addWidget(self.elbow_length_input, 2, 1)
        dimensions_layout.addWidget(QLabel("cm"), 2, 2)
        
        # Gripper length
        dimensions_layout.addWidget(QLabel("Gripper Length:"), 3, 0)
        self.gripper_length_input = QLineEdit(str(self.robot.gripper_length))
        self.gripper_length_input.setMaximumWidth(50)
        dimensions_layout.addWidget(self.gripper_length_input, 3, 1)
        dimensions_layout.addWidget(QLabel("cm"), 3, 2)
        
        # Apply dimensions button
        apply_dimensions_button = QPushButton("Apply Dimensions")
        apply_dimensions_button.clicked.connect(self.apply_dimensions)
        dimensions_layout.addWidget(apply_dimensions_button, 4, 0, 1, 3)
        
        layout.addWidget(dimensions_group)
        
        # Add spacer to push everything to the top
        layout.addStretch()
    
    def setup_multiple_solutions_tab(self):
        """Setup the tab for visualizing and comparing multiple IK solutions"""
        layout = QVBoxLayout(self.solutions_tab)
        
        # Solution navigation
        nav_group = QGroupBox("Solution Navigation")
        nav_layout = QGridLayout()
        nav_group.setLayout(nav_layout)
        
        self.solution_count_label = QLabel("No solutions available")
        nav_layout.addWidget(self.solution_count_label, 0, 0, 1, 3)
        
        prev_button = QPushButton("Previous Solution")
        prev_button.clicked.connect(self.previous_solution)
        nav_layout.addWidget(prev_button, 1, 0)
        
        self.current_solution_label = QLabel("0/0")
        nav_layout.addWidget(self.current_solution_label, 1, 1)
        
        next_button = QPushButton("Next Solution")
        next_button.clicked.connect(self.next_solution)
        nav_layout.addWidget(next_button, 1, 2)
        
        layout.addWidget(nav_group)
        
        # Current solution display
        solution_group = QGroupBox("Current Solution")
        solution_layout = QGridLayout()
        solution_group.setLayout(solution_layout)
        
        solution_layout.addWidget(QLabel("Base:"), 0, 0)
        self.base_multi_label = QLabel("0.00°")
        solution_layout.addWidget(self.base_multi_label, 0, 1)
        
        solution_layout.addWidget(QLabel("Shoulder:"), 1, 0)
        self.shoulder_multi_label = QLabel("0.00°")
        solution_layout.addWidget(self.shoulder_multi_label, 1, 1)
        
        solution_layout.addWidget(QLabel("Elbow:"), 2, 0)
        self.elbow_multi_label = QLabel("0.00°")
        solution_layout.addWidget(self.elbow_multi_label, 2, 1)
        
        solution_layout.addWidget(QLabel("Wrist:"), 3, 0)
        self.wrist_multi_label = QLabel("0.00°")
        solution_layout.addWidget(self.wrist_multi_label, 3, 1)
        
        solution_layout.addWidget(QLabel("Configuration:"), 4, 0)
        self.config_multi_label = QLabel("None")
        solution_layout.addWidget(self.config_multi_label, 4, 1)
        
        solution_layout.addWidget(QLabel("Error:"), 5, 0)
        self.error_multi_label = QLabel("0.00 cm")
        solution_layout.addWidget(self.error_multi_label, 5, 1)
        
        # Apply current solution button
        apply_multi_button = QPushButton("Apply This Solution")
        apply_multi_button.clicked.connect(self.apply_multi_solution)
        solution_layout.addWidget(apply_multi_button, 6, 0, 1, 2)
        
        layout.addWidget(solution_group)
        
        # Animation controls
        animation_group = QGroupBox("Path Animation")
        animation_layout = QGridLayout()
        animation_group.setLayout(animation_layout)
        
        # Path type selection
        animation_layout.addWidget(QLabel("Path Type:"), 0, 0)
        self.path_type_combo = QComboBox()
        self.path_type_combo.addItems(["Straight Line", "Half Circle Up", "Half Circle Down", "Half Figure-8"])
        animation_layout.addWidget(self.path_type_combo, 0, 1, 1, 2)
        
        # Animation controls
        animation_layout.addWidget(QLabel("Animation Speed:"), 1, 0)
        self.animation_speed_slider = QSlider(Qt.Horizontal)
        self.animation_speed_slider.setRange(1, 10)
        self.animation_speed_slider.setValue(5)
        animation_layout.addWidget(self.animation_speed_slider, 1, 1, 1, 2)
        
        # Animation control buttons
        start_button = QPushButton("Start Animation")
        start_button.clicked.connect(self.start_animation)
        animation_layout.addWidget(start_button, 2, 0)
        
        stop_button = QPushButton("Stop Animation")
        stop_button.clicked.connect(self.stop_animation)
        animation_layout.addWidget(stop_button, 2, 1)
        
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_animation)
        animation_layout.addWidget(reset_button, 2, 2)
        
        # Add a new row for the print path button
        print_path_button = QPushButton("Print Path Information")
        print_path_button.clicked.connect(self.print_path_information)
        animation_layout.addWidget(print_path_button, 3, 0, 1, 3)
        
        layout.addWidget(animation_group)
        
        # Solution export and comparison
        export_group = QGroupBox("Export & Analysis")
        export_layout = QGridLayout()
        export_group.setLayout(export_layout)
        
        export_button = QPushButton("Export Solutions to CSV")
        export_button.clicked.connect(self.export_solutions)
        export_layout.addWidget(export_button, 0, 0, 1, 2)
        
        visualize_button = QPushButton("Visualize All Solutions")
        visualize_button.clicked.connect(self.visualize_all_solutions)
        export_layout.addWidget(visualize_button, 1, 0, 1, 2)
        
        # Add new button for real arm path execution
        self.follow_path_real_button = QPushButton("Follow Path on Real Arm")
        self.follow_path_real_button.clicked.connect(self.trigger_real_arm_path_execution)
        animation_layout.addWidget(self.follow_path_real_button, animation_layout.rowCount(), 0, 1, animation_layout.columnCount()) # Add to new row

        # Add a status label for real arm path execution
        self.path_execution_status_label = QLabel("Real Arm Status: Idle")
        animation_layout.addWidget(self.path_execution_status_label, animation_layout.rowCount(), 0, 1, animation_layout.columnCount())
        
        
        layout.addWidget(export_group)
        
        # Add spacer to push everything to the top
        layout.addStretch()
    
    def setup_3d_plot(self, layout):
        """Setup the 3D plot for visualizing the robot arm"""
        # Create a figure and 3D axes
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 2)
        
        # Create 3D axes
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_zlabel('Z (cm)')
        
        # Set axis limits
        max_reach = self.robot.base_height + self.robot.shoulder_length + self.robot.elbow_length + self.robot.gripper_length
        self.ax.set_xlim(-max_reach, max_reach)
        self.ax.set_ylim(-max_reach, max_reach)
        self.ax.set_zlim(0, max_reach)
        
        # Equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])
        
        # Initialize plot elements
        self.arm_line, = self.ax.plot([], [], [], 'o-', linewidth=2, markersize=6)
        self.end_effector_point, = self.ax.plot([], [], [], 'ro', markersize=8)
        
        # Initialize gripper lines
        self.left_gripper, = self.ax.plot([], [], [], 'g-', linewidth=1.5)
        self.right_gripper, = self.ax.plot([], [], [], 'g-', linewidth=1.5)
        
        # Initialize target point and path
        self.target_point, = self.ax.plot([], [], [], 'yo', markersize=8)
        self.path_line, = self.ax.plot([], [], [], 'c-', linewidth=1, alpha=0.5)
        
        # Add a title
        self.ax.set_title('4DoF Robot Arm Visualization')
    
    def update_joint_angles(self):
        """Update the robot arm joint angles from the sliders"""
        # Get values from sliders and convert to radians
        base_angle = np.radians(self.base_slider.value())
        shoulder_angle = np.radians(self.shoulder_slider.value())
        elbow_angle = np.radians(self.elbow_slider.value())
        wrist_angle = np.radians(self.wrist_slider.value())
        gripper_state = self.gripper_slider.value() / 100.0
        
        # Update the robot arm
        self.robot.set_joint_angles(base_angle, shoulder_angle, elbow_angle, wrist_angle, gripper_state)
        
        # Update the value labels
        self.base_value_label.setText(f"{self.base_slider.value()}°")
        self.shoulder_value_label.setText(f"{self.shoulder_slider.value()}°")
        self.elbow_value_label.setText(f"{self.elbow_slider.value()}°")
        self.wrist_value_label.setText(f"{self.wrist_slider.value()}°")
        
        if gripper_state < 0.1:
            self.gripper_value_label.setText("Closed")
        elif gripper_state > 0.9:
            self.gripper_value_label.setText("Open")
        else:
            self.gripper_value_label.setText(f"{int(gripper_state * 100)}%")
        
        # Update the position labels
        self.x_position_label.setText(f"{self.robot.end_effector[0]:.2f} cm")
        self.y_position_label.setText(f"{self.robot.end_effector[1]:.2f} cm")
        self.z_position_label.setText(f"{self.robot.end_effector[2]:.2f} cm")
                # --- Add Servo Command Logic ---
        # Get integer degrees directly from sliders for sending
        base_deg = self.base_slider.value()
        shoulder_deg = self.shoulder_slider.value()
        elbow_deg = self.elbow_slider.value()
        wrist_deg = self.wrist_slider.value()
        gripper_val = self.gripper_slider.value() # 0-100

        # Map GUI labels/sliders to servo names used in Pi script
        # IMPORTANT: Convert angles to integers.
        # You might need to adjust ranges/offsets based on your physical setup!
        angles_to_send = {
            "base": int(base_deg),
            "shoulder": int(shoulder_deg), # Corresponds to 'arm1' in your screenshot
            "elbow": int(elbow_deg),       # Corresponds to 'arm2' in your screenshot
            "wrist": int(wrist_deg),       # Corresponds to 'wrist' in your screenshot
            # Gripper state needs mapping: Slider 0-100 -> Angle (e.g., 0=closed, 90=open)
            # Example mapping: 0 slider -> 0 degrees, 100 slider -> 90 degrees
            "gripper_wrist": int(np.interp(gripper_val, [0, 100], [20, 60])) # Map 0-100 to 0-90 degrees. Adjust [0, 90] as needed!
        }
        # Send these angles to the Pi using the threaded function
        self.send_angles_to_pi(angles_to_send)
        # --- End Servo Command Logic ---

        # Note: self.update_plot() is called by the QTimer, so no need to call it here explicitly unless you want instant visual feedback *before* the timer tick.
   
    
    def apply_offsets(self):
        """Apply the calibration offsets"""
        try:
            base_offset = np.radians(float(self.base_offset_input.text()))
            shoulder_offset = np.radians(float(self.shoulder_offset_input.text()))
            elbow_offset = np.radians(float(self.elbow_offset_input.text()))
            wrist_offset = np.radians(float(self.wrist_offset_input.text()))
            
            self.robot.set_offsets(base_offset, shoulder_offset, elbow_offset, wrist_offset)
            self.update_joint_angles()  # Update to reflect changes
        except ValueError:
            print("Invalid offset values. Please enter numeric values.")
    
    def apply_dimensions(self):
        """Apply the arm dimension changes"""
        try:
            base_height = float(self.base_height_input.text())
            shoulder_length = float(self.shoulder_length_input.text())
            elbow_length = float(self.elbow_length_input.text())
            gripper_length = float(self.gripper_length_input.text())
            
            self.robot.set_dimensions(base_height, shoulder_length, elbow_length, gripper_length)
            
            # Update plot limits
            max_reach = base_height + shoulder_length + elbow_length + gripper_length
            self.ax.set_xlim(-max_reach, max_reach)
            self.ax.set_ylim(-max_reach, max_reach)
            self.ax.set_zlim(0, max_reach)
            
            self.update_joint_angles()  # Update to reflect changes
        except ValueError:
            print("Invalid dimension values. Please enter numeric values.")
    
    def solve_inverse_kinematics(self):
        """Solve inverse kinematics for the target position using specified configuration"""
        try:
            target_x = float(self.x_target_input.text())
            target_y = float(self.y_target_input.text())
            target_z = float(self.z_target_input.text())
            
            # Get configuration preference
            config_option = self.config_combobox.currentText()
            if config_option == "Elbow Up":
                config = "elbow_up"
            elif config_option == "Elbow Down":
                config = "elbow_down"
            else:
                config = None  # Auto (best)
            
            # Display the configuration being used
            config_display = config if config else "Auto (Best)"
            self.ik_status_label.setText(f"Solving with {config_display} configuration...")
            
            # Get orientation if enabled
            orientation = None
            if self.orientation_enabled.isChecked():
                orientation = np.radians(float(self.orientation_input.text()))
            
            # Get error threshold and max iterations
            error_threshold = float(self.error_threshold_input.text())
            max_iterations = int(self.max_iterations_input.text())
            
            # Solve IK with the specified configuration
            success, solution = self.robot.inverse_kinematics(
                target_x, target_y, target_z,
                config=config, 
                orientation=orientation,
                error_threshold=error_threshold,
                max_iterations=max_iterations
            )
            
            # Update target visualization
            self.target_point.set_data([target_x], [target_y])
            self.target_point.set_3d_properties([target_z])
            
            if success:
                base_angle, shoulder_angle, elbow_angle, wrist_angle = solution
                
                # Convert solution to degrees for display
                base_deg = np.degrees(base_angle)
                shoulder_deg = np.degrees(shoulder_angle)
                elbow_deg = np.degrees(elbow_angle)
                wrist_deg = np.degrees(wrist_angle)
                
                # Update solution labels
                self.base_solution_label.setText(f"{base_deg:.2f}°")
                self.shoulder_solution_label.setText(f"{shoulder_deg:.2f}°")
                self.elbow_solution_label.setText(f"{elbow_deg:.2f}°")
                self.wrist_solution_label.setText(f"{wrist_deg:.2f}°")
                
                # Determine configuration
                if elbow_angle < 0:
                    self.config_solution_label.setText("Elbow Up")
                else:
                    self.config_solution_label.setText("Elbow Down")
                
                # Calculate error
                end_pos = self.robot.end_effector
                error = np.linalg.norm(np.array([target_x, target_y, target_z]) - end_pos)
                self.error_solution_label.setText(f"{error:.4f} cm")
                
                self.ik_status_label.setText("Solution found")
            else:
                self.ik_status_label.setText("No valid solution")
            
        except ValueError as e:
            self.ik_status_label.setText(f"Invalid input: {str(e)}")
            print(f"Error: {str(e)}")
    
    def find_all_solutions(self):
        """Find all valid IK solutions for the target position and store them"""
        try:
            target_x = float(self.x_target_input.text())
            target_y = float(self.y_target_input.text())
            target_z = float(self.z_target_input.text())
            
            # Get error threshold and max iterations
            error_threshold = float(self.error_threshold_input.text())
            max_iterations = int(self.max_iterations_input.text())
            
            # Get orientation if enabled
            orientation = None
            if self.orientation_enabled.isChecked():
                orientation = np.radians(float(self.orientation_input.text()))
            
            # Find all solutions
            success, solutions = self.robot.inverse_kinematics(
                target_x, target_y, target_z,
                orientation=orientation,
                error_threshold=error_threshold,
                max_iterations=max_iterations,
                return_all=True
            )
            
            # Update target visualization
            self.target_point.set_data([target_x], [target_y])
            self.target_point.set_3d_properties([target_z])
            
            if success:
                # Store the solutions with additional info
                self.solutions = []
                for solution in solutions:
                    base_angle, shoulder_angle, elbow_angle, wrist_angle = solution
                    
                    # Set the robot to this solution to check error
                    self.robot.set_joint_angles(base_angle, shoulder_angle, elbow_angle, wrist_angle)
                    end_pos = self.robot.end_effector
                    error = np.linalg.norm(np.array([target_x, target_y, target_z]) - end_pos)
                    
                    # Determine configuration
                    config = "Elbow Up" if elbow_angle < 0 else "Elbow Down"
                    
                    # Store solution data
                    self.solutions.append({
                        'angles': solution,
                        'error': error,
                        'config': config,
                        'target': [target_x, target_y, target_z]
                    })
                
                # Reset to the first solution
                self.current_solution_index = 0
                self.update_multiple_solutions_display()
                
                # Switch to the multiple solutions tab
                self.tabs.setCurrentIndex(2)
                
                self.ik_status_label.setText(f"Found {len(self.solutions)} solutions")
            else:
                self.ik_status_label.setText("No valid solutions found")
                self.solutions = []
                self.update_multiple_solutions_display()
            
        except ValueError as e:
            self.ik_status_label.setText(f"Invalid input: {str(e)}")
            print(f"Error: {str(e)}")
    
    def update_multiple_solutions_display(self):
        """Update the display for the multiple solutions tab"""
        if not self.solutions:
            self.solution_count_label.setText("No solutions available")
            self.current_solution_label.setText("0/0")
            self.base_multi_label.setText("0.00°")
            self.shoulder_multi_label.setText("0.00°")
            self.elbow_multi_label.setText("0.00°")
            self.wrist_multi_label.setText("0.00°")
            self.config_multi_label.setText("None")
            self.error_multi_label.setText("0.00 cm")
            return
        
        # Update navigation display
        total_solutions = len(self.solutions)
        self.solution_count_label.setText(f"Found {total_solutions} solutions")
        self.current_solution_label.setText(f"{self.current_solution_index + 1}/{total_solutions}")
        
        # Get current solution
        solution = self.solutions[self.current_solution_index]
        base_angle, shoulder_angle, elbow_angle, wrist_angle = solution['angles']
        
        # Update solution display
        self.base_multi_label.setText(f"{np.degrees(base_angle):.2f}°")
        self.shoulder_multi_label.setText(f"{np.degrees(shoulder_angle):.2f}°")
        self.elbow_multi_label.setText(f"{np.degrees(elbow_angle):.2f}°")
        self.wrist_multi_label.setText(f"{np.degrees(wrist_angle):.2f}°")
        self.config_multi_label.setText(solution['config'])
        self.error_multi_label.setText(f"{solution['error']:.4f} cm")
        
        # Update the robot visualization
        self.robot.set_joint_angles(base_angle, shoulder_angle, elbow_angle, wrist_angle)
        self.update_plot()
    
    def previous_solution(self):
        """Navigate to the previous solution"""
        if not self.solutions:
            return
        
        total_solutions = len(self.solutions)
        self.current_solution_index = (self.current_solution_index - 1) % total_solutions
        self.update_multiple_solutions_display()
    
    def next_solution(self):
        """Navigate to the next solution"""
        if not self.solutions:
            return
        
        total_solutions = len(self.solutions)
        self.current_solution_index = (self.current_solution_index + 1) % total_solutions
        self.update_multiple_solutions_display()
    
    def apply_multi_solution(self):
        """Apply the currently displayed multiple solution"""
        if not self.solutions:
            return
        
        # Get current solution
        solution = self.solutions[self.current_solution_index]
        base_angle, shoulder_angle, elbow_angle, wrist_angle = solution['angles']
        
        # Set the robot arm angles
        self.robot.set_joint_angles(base_angle, shoulder_angle, elbow_angle, wrist_angle)
        
        # Update the sliders to match
        self.base_slider.setValue(int(np.degrees(base_angle)))
        self.shoulder_slider.setValue(int(np.degrees(shoulder_angle)))
        self.elbow_slider.setValue(int(np.degrees(elbow_angle)))
        self.wrist_slider.setValue(int(np.degrees(wrist_angle)))
        
        # Switch to the forward kinematics tab
        self.tabs.setCurrentIndex(0)
    
    def apply_ik_solution(self):
        """Apply the inverse kinematics solution to the robot arm"""
        try:
            # Get the current solution from the labels
            base_deg = float(self.base_solution_label.text().replace('°', ''))
            shoulder_deg = float(self.shoulder_solution_label.text().replace('°', ''))
            elbow_deg = float(self.elbow_solution_label.text().replace('°', ''))
            wrist_deg = float(self.wrist_solution_label.text().replace('°', ''))
            
            # Map GUI labels/sliders to servo names used in Pi script
            # IMPORTANT: Convert angles to integers as requested.
            # You might need to adjust ranges/offsets based on your physical setup!
            angles_to_send = {
                "base": int(base_deg),
                "shoulder": int(shoulder_deg), # Corresponds to 'arm1' in your screenshot
                "elbow": int(elbow_deg),       # Corresponds to 'arm2' in your screenshot
                "wrist": int(wrist_deg),       # Corresponds to 'wrist' in your screenshot
                # Gripper state needs mapping: Slider 0-100 -> Angle (e.g., 0=closed, 90=open)
                # Example mapping: 0 slider -> 0 degrees, 100 slider -> 90 degrees
                "gripper_wrist": int(np.interp(self.gripper_slider.value(), [0, 100], [0, 90])) # Map 0-100 to 0-90 degrees. Adjust [0, 90] as needed!
            }
            self.send_angles_to_pi(angles_to_send)
            # --- End Servo Command Logic ---
            
            # Convert back to radians for the robot model
            base_rad = np.radians(base_deg)
            shoulder_rad = np.radians(shoulder_deg)
            elbow_rad = np.radians(elbow_deg)
            wrist_rad = np.radians(wrist_deg)
            
            # Update the robot model first - direct assignment ensures accuracy
            self.robot.set_joint_angles(base_rad, shoulder_rad, elbow_rad, wrist_rad)
            
            # Update the sliders
            self.base_slider.setValue(int(base_deg))
            self.shoulder_slider.setValue(int(shoulder_deg))
            self.elbow_slider.setValue(int(elbow_deg))
            self.wrist_slider.setValue(int(wrist_deg))
            
            # Update position display
            end_pos = self.robot.end_effector
            self.x_position_label.setText(f"{end_pos[0]:.2f} cm")
            self.y_position_label.setText(f"{end_pos[1]:.2f} cm")
            self.z_position_label.setText(f"{end_pos[2]:.2f} cm")
            
            # Switch to the forward kinematics tab
            self.tabs.setCurrentIndex(0)
            
            # Update the visualization
            self.update_plot()
            
            print(f"Applied solution: Base={base_deg:.2f}°, Shoulder={shoulder_deg:.2f}°, Elbow={elbow_deg:.2f}°, Wrist={wrist_deg:.2f}°")
            print(f"End effector position: ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f})")
            
        except ValueError as e:
            print(f"Error applying solution: {str(e)}")
    
    def get_movement_step_size(self):
        """Get the movement step size from the active tab's input field with error checking"""
        try:
            # Always use the Forward Kinematics tab's step size input for directional controls
            step_size = float(self.fk_step_size_input.text())
            
            # Validate step size and provide reasonable limits
            if step_size <= 0:
                print("Warning: Step size must be positive, using default 0.5 cm")
                return 0.5  # Default if negative or zero
            
            # Cap the step size to prevent too large movements
            if step_size > 5.0:
                print(f"Warning: Step size {step_size} cm is too large, capping at 5.0 cm")
                return 5.0
                
            return step_size
        except (ValueError, AttributeError):
            print("Warning: Invalid step size input, using default 0.5 cm")
            return 0.5  # Default if invalid
    
    # Removed move_end_effector method - using direct IK instead
    
    def move_end_effector_up(self):
        """Move the end effector upward in Z direction"""
        # Get step size with validation
        step = self.get_movement_step_size()
        print(f"Moving up {step} cm")
        
        # Get current end effector position directly from robot
        current_x, current_y, current_z = self.robot.end_effector
        
        # Calculate target position
        target_x = current_x
        target_y = current_y
        target_z = current_z + step
        
        # Apply inverse kinematics
        self.solve_to_position(target_x, target_y, target_z)
    
    def move_end_effector_down(self):
        """Move the end effector downward in Z direction"""
        step = self.get_movement_step_size()
        print(f"Moving down {step} cm")
        
        # Get current end effector position
        current_x, current_y, current_z = self.robot.end_effector
        
        # Calculate target position
        target_x = current_x
        target_y = current_y
        target_z = current_z - step
        
        # Apply inverse kinematics
        self.solve_to_position(target_x, target_y, target_z)
    
    def move_end_effector_left(self):
        """Move the end effector left in Y direction"""
        step = self.get_movement_step_size()
        print(f"Moving left {step} cm")
        
        # Get current end effector position
        current_x, current_y, current_z = self.robot.end_effector
        
        # Calculate target position
        target_x = current_x
        target_y = current_y + step
        target_z = current_z
        
        # Apply inverse kinematics
        self.solve_to_position(target_x, target_y, target_z)
    
    def move_end_effector_right(self):
        """Move the end effector right in Y direction"""
        step = self.get_movement_step_size()
        print(f"Moving right {step} cm")
        
        # Get current end effector position
        current_x, current_y, current_z = self.robot.end_effector
        
        # Calculate target position
        target_x = current_x
        target_y = current_y - step
        target_z = current_z
        
        # Apply inverse kinematics
        self.solve_to_position(target_x, target_y, target_z)
    
    def move_end_effector_forward(self):
        """Move the end effector forward in X direction"""
        step = self.get_movement_step_size()
        print(f"Moving forward {step} cm")
        
        # Get current end effector position
        current_x, current_y, current_z = self.robot.end_effector
        
        # Calculate target position
        target_x = current_x + step
        target_y = current_y
        target_z = current_z
        
        # Apply inverse kinematics
        self.solve_to_position(target_x, target_y, target_z)
    
    def move_end_effector_backward(self):
        """Move the end effector backward in X direction"""
        step = self.get_movement_step_size()
        print(f"Moving backward {step} cm")
        
        # Get current end effector position
        current_x, current_y, current_z = self.robot.end_effector
        
        # Calculate target position
        target_x = current_x - step
        target_y = current_y
        target_z = current_z
        
        # Apply inverse kinematics
        self.solve_to_position(target_x, target_y, target_z)
    
    def solve_to_position(self, target_x, target_y, target_z):
        """
        Solve inverse kinematics to a specified target position and apply solution.
        """
        # Update the target input fields
        self.x_target_input.setText(f"{target_x:.2f}")
        self.y_target_input.setText(f"{target_y:.2f}")
        self.z_target_input.setText(f"{target_z:.2f}")
        
        # Try to maintain current configuration
        current_config = "elbow_up" if self.robot.elbow_angle < 0 else "elbow_down"
        
        # Get configuration preference
        config_option = self.config_combobox.currentText()
        if config_option == "Elbow Up":
            config = "elbow_up"
        elif config_option == "Elbow Down":
            config = "elbow_down"
        else:
            # When set to Auto, try to maintain the current configuration
            config = current_config
        
        try:
            # Use increased iterations and tighter error threshold for precision
            error_threshold = 0.1  # Reduced error threshold for more precision
            max_iterations = 1000    # Increased iterations for better convergence
            
            print(f"Solving for position ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) with {max_iterations} iterations and error threshold {error_threshold}")
            
            # Solve IK with improved parameters
            success, solution = self.robot.inverse_kinematics(
                target_x, target_y, target_z,
                config=config,
                error_threshold=error_threshold,
                max_iterations=max_iterations
            )
            
            if success:
                # Update target visualization
                self.target_point.set_data([target_x], [target_y])
                self.target_point.set_3d_properties([target_z])
                
                # Apply the solution
                base_angle, shoulder_angle, elbow_angle, wrist_angle = solution
                self.robot.set_joint_angles(base_angle, shoulder_angle, elbow_angle, wrist_angle)
                
                # Convert solution to degrees for display
                base_deg = np.degrees(base_angle)
                shoulder_deg = np.degrees(shoulder_angle)
                elbow_deg = np.degrees(elbow_angle)
                wrist_deg = np.degrees(wrist_angle)
                
                # Update solution labels
                self.base_solution_label.setText(f"{base_deg:.2f}°")
                self.shoulder_solution_label.setText(f"{shoulder_deg:.2f}°")
                self.elbow_solution_label.setText(f"{elbow_deg:.2f}°")
                self.wrist_solution_label.setText(f"{wrist_deg:.2f}°")
                
                # Update sliders
                self.base_slider.setValue(int(base_deg))
                self.shoulder_slider.setValue(int(shoulder_deg))
                self.elbow_slider.setValue(int(elbow_deg))
                self.wrist_slider.setValue(int(wrist_deg))
                
                # Calculate error
                end_pos = self.robot.end_effector
                error = np.linalg.norm(np.array([target_x, target_y, target_z]) - end_pos)
                
                # Update error, position, and configuration displays
                self.error_solution_label.setText(f"{error:.4f} cm")
                self.x_position_label.setText(f"{end_pos[0]:.2f} cm")
                self.y_position_label.setText(f"{end_pos[1]:.2f} cm")
                self.z_position_label.setText(f"{end_pos[2]:.2f} cm")
                
                if elbow_angle < 0:
                    self.config_solution_label.setText("Elbow Up")
                else:
                    self.config_solution_label.setText("Elbow Down")
                
                # Update the plot
                self.update_plot()
                
                print(f"Moved to ({end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}) with error {error:.4f} cm")
            else:
                print(f"Failed to reach position ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
        except Exception as e:
            print(f"Error applying inverse kinematics: {str(e)}")
    
    def sync_step_size_inputs(self, text):
        """Keep the step size inputs in both tabs synchronized"""
        # Determine which input was changed
        sender = self.sender()
        
        try:
            # Validate the input is a valid number first
            value = float(text)
            
            # Update the step size input in the IK tab if it exists
            if hasattr(self, 'step_size_input'):
                # Block signals to prevent recursive loop
                self.step_size_input.blockSignals(True)
                self.step_size_input.setText(text)
                self.step_size_input.blockSignals(False)
                
            # If this was called from somewhere other than the FK input, update that one
            if sender != self.fk_step_size_input and hasattr(self, 'fk_step_size_input'):
                self.fk_step_size_input.blockSignals(True)
                self.fk_step_size_input.setText(text)
                self.fk_step_size_input.blockSignals(False)
                
        except ValueError:
            # Not a valid number, don't propagate it
            pass
    
    def start_animation(self):
        """Start the path animation using the selected configuration"""
        if self.animation_running:
            return
        
        # Get the selected configuration for the animation
        config_idx = self.config_combobox.currentIndex()
        if config_idx == 0:
            # "Auto (Best)" means no fixed configuration
            self.animation_config = None
        elif config_idx == 1:
            # "Elbow Up" configuration
            self.animation_config = 'elbow_up'
        else:
            # "Elbow Down" configuration
            self.animation_config = 'elbow_down'
            
        # Generate the path based on selected type and pre-calculate solutions
        path_type = self.path_type_combo.currentText()
        path_valid = self.generate_animation_path(path_type)
        
        # Ensure path is valid and solutions were found
        if not path_valid or not self.animation_path:
            # Show error message
            self.solution_count_label.setText(f"Cannot animate: Some points cannot be reached with {self.animation_config if self.animation_config else 'Auto'} configuration")
            return
        
        # Start animation
        self.animation_running = True
        self.animation_index = 0
        
        # Show the selected configuration for this animation
        config_name = self.animation_config if self.animation_config else "Auto (Best)"
        self.solution_count_label.setText(f"Animation running with: {config_name} configuration")
        
        # Start at the beginning of the path
        self.perform_animation_step()
    
    def stop_animation(self):
        """Stop the path animation"""
        self.animation_running = False
    
    def reset_animation(self):
        """Reset the animation to the beginning"""
        self.animation_running = False
        self.animation_index = 0
        self.animation_path = []
        
        # Reset the path visualization
        self.path_line.set_data([], [])
        self.path_line.set_3d_properties([])
        self.target_point.set_data([], [])
        self.target_point.set_3d_properties([])
        
        self.update_plot()
    
    def generate_animation_path(self, path_type):
        """Generate a path for animation based on path type and verify solutions"""
        # Clear previous path
        self.animation_path = []
        self.animation_solutions = []
        
        # Number of points in the path
        num_points = 50
        
        # Get initial and target points from input fields
        try:
            # Get initial point (use current position if not specified)
            init_x = float(self.x_position_label.text().split()[0])
            init_y = float(self.y_position_label.text().split()[0])
            init_z = float(self.z_position_label.text().split()[0])
            
            # Get target point from IK tab
            target_x = float(self.x_target_input.text())
            target_y = float(self.y_target_input.text())
            target_z = float(self.z_target_input.text())
            
            # Update status
            self.solution_count_label.setText(f"Generating path from ({init_x:.1f}, {init_y:.1f}, {init_z:.1f}) to ({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
            
            # Draw the initial and target points
            self.path_line.set_data([init_x, target_x], [init_y, target_y])
            self.path_line.set_3d_properties([init_z, target_z])
            self.target_point.set_data([target_x], [target_y])
            self.target_point.set_3d_properties([target_z])
            self.canvas.draw()
            
        except ValueError:
            # Default values if no valid input
            init_x, init_y, init_z = 0, 0, 15
            target_x, target_y, target_z = 15, 0, 15
            self.solution_count_label.setText("Using default path (no valid input points)")
        
        # Calculate path parameters based on initial and target points
        dx = target_x - init_x
        dy = target_y - init_y
        dz = target_z - init_z
        
        # Distance between points in XY plane
        distance_xy = np.sqrt(dx**2 + dy**2)
        
        # Total 3D distance
        distance_3d = np.sqrt(distance_xy**2 + dz**2)
        
        # Direction vector from initial to target in XY plane (normalized)
        if distance_xy > 0.001:  # Avoid division by zero
            dir_x, dir_y = dx/distance_xy, dy/distance_xy
        else:
            dir_x, dir_y = 0, 1  # Default direction if points have same XY position
        
        # Perpendicular vector in XY plane (for arc path)
        perp_x, perp_y = -dir_y, dir_x
        
        # Calculate midpoint
        mid_x = (init_x + target_x) / 2
        mid_y = (init_y + target_y) / 2
        mid_z = (init_z + target_z) / 2
        
        # Generate the path points based on selected type
        start_point = (init_x, init_y, init_z)
        end_point = (target_x, target_y, target_z)
        
        # Use the PathPlanner to generate the paths
        if path_type == "Straight Line":
            self.animation_path = PathPlanner.generate_path('straight', start_point, end_point, num_points)
        
        elif path_type == "Half Circle Up":
            # Use the half circle up path
            self.animation_path = PathPlanner.generate_path('half_circle_up', start_point, end_point, num_points, height_factor=0.5)
        
        elif path_type == "Half Circle Down":
            # Use the half circle down path
            self.animation_path = PathPlanner.generate_path('half_circle_down', start_point, end_point, num_points, height_factor=0.5)
        
        elif path_type == "Half Figure-8":
            # Use the half figure-8 path
            self.animation_path = PathPlanner.generate_path('half_figure8', start_point, end_point, num_points, height_factor=0.4)
                
        elif path_type == "Circle":
            # Full circle centered between initial and target points
            radius = distance_xy / 2
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = mid_x + radius * np.cos(angle)
                y = mid_y + radius * np.sin(angle)
                z = mid_z + 2.0 * np.sin(angle)
                self.animation_path.append((x, y, z))
                
        elif path_type == "Complete Figure-8":
            # Full figure-8 pattern
            radius = distance_xy / 3
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = mid_x + radius * np.sin(angle)
                y = mid_y + radius * np.sin(angle) * np.cos(angle)
                z = mid_z + 4.0 * np.sin(angle * 0.5)
                self.animation_path.append((x, y, z))
        
        # Visualize the path
        path_x = [p[0] for p in self.animation_path]
        path_y = [p[1] for p in self.animation_path]
        path_z = [p[2] for p in self.animation_path]
        
        self.path_line.set_data(path_x, path_y)
        self.path_line.set_3d_properties(path_z)
        
        # Pre-calculate IK solutions for all points
        self.solution_count_label.setText("Calculating solutions for path...")
        self.canvas.draw()  # Force update to show the message
        
        # Check all points for solutions with increasing tolerance if needed
        unreachable_points = []
        high_error_points = []
        
        # Use the configuration from dropdown
        config_option = self.config_combobox.currentText()
        if config_option == "Elbow Up":
            self.animation_config = "elbow_up"
        elif config_option == "Elbow Down":
            self.animation_config = "elbow_down"
        else:
            self.animation_config = None  # Auto (best)
        
        # Try with increasing error thresholds
        thresholds = [0.1, 0.1, 0.1, 1.0]
        max_iterations = 100
        
        for i, point in enumerate(self.animation_path):
            x, y, z = point
            
            # Try different error thresholds
            solved = False
            best_solution = None
            best_error = float('inf')
            
            for threshold in thresholds:
                # Try to solve IK with current threshold
                success, solution = self.robot.inverse_kinematics(
                    x, y, z, 
                    config=self.animation_config,
                    error_threshold=threshold,
                    max_iterations=max_iterations
                )
                
                if success:
                    # Calculate the actual error
                    base_angle, shoulder_angle, elbow_angle, wrist_angle = solution
                    
                    # Save original state
                    original_angles = (self.robot.base_angle, self.robot.shoulder_angle, 
                                       self.robot.elbow_angle, self.robot.wrist_angle)
                    
                    # Set angles to check error
                    self.robot.set_joint_angles(*solution)
                    end_pos = self.robot.end_effector
                    error = np.linalg.norm(np.array([x, y, z]) - end_pos)
                    
                    # Restore original state
                    self.robot.set_joint_angles(*original_angles)
                    
                    if error < best_error:
                        best_error = error
                        best_solution = solution
                        
                        # If error is acceptable, stop trying higher thresholds
                        if error < 0.5:
                            solved = True
                            break
            
            if best_solution:
                self.animation_solutions.append(best_solution)
                if best_error > 0.5:
                    high_error_points.append((i, best_error))
            else:
                unreachable_points.append(i)
                # Add a placeholder so the indexes match
                self.animation_solutions.append(None)
        
        # Update the display with results
        if unreachable_points:
            message = f"Warning: {len(unreachable_points)} points cannot be reached with {config_option} configuration."
            self.solution_count_label.setText(message)
            print(message)
            print(f"Unreachable at indices: {unreachable_points}")
            return False
        elif high_error_points:
            message = f"Path generated with {len(high_error_points)} high-error points."
            self.solution_count_label.setText(message)
            print(message)
            print(f"High error points: {high_error_points}")
            return True
        else:
            self.solution_count_label.setText(f"Path generated successfully with {len(self.animation_path)} points.")
            return True
    
    def perform_animation_step(self):
        """Perform a single step of the animation using pre-calculated solutions"""
        if not self.animation_running or not self.animation_path or not self.animation_solutions:
            return
        
        # Get the current point and pre-calculated solution
        x, y, z = self.animation_path[self.animation_index]
        solution = self.animation_solutions[self.animation_index]
        
        # Update target visualization - always show the final target point
        final_target = self.animation_path[-1]
        self.target_point.set_data([final_target[0]], [final_target[1]])
        self.target_point.set_3d_properties([final_target[2]])
        
        # Set the robot to the pre-calculated solution
        if solution:
            base_angle, shoulder_angle, elbow_angle, wrist_angle = solution
            self.robot.set_joint_angles(base_angle, shoulder_angle, elbow_angle, wrist_angle, 0.5)
            
            # Determine and display the current configuration
            if elbow_angle < 0:
                config_name = "Elbow-Up"
            else:
                config_name = "Elbow-Down"
            
            # Update configuration status
            self.config_multi_label.setText(config_name)
            
            # Calculate and display error
            error = np.linalg.norm(np.array([x, y, z]) - self.robot.end_effector)
            self.error_multi_label.setText(f"{error:.4f} cm")
            
            # Display point index
            current_point = self.animation_index + 1
            total_points = len(self.animation_path)
            self.current_solution_label.setText(f"Point {current_point}/{total_points}")
            
            # Update the display
            self.update_plot()
        else:
            # This should not happen since we pre-verified all solutions,
            # but handle it just in case
            self.error_multi_label.setText("Cannot reach this point")
        
        # Move to the next point
        self.animation_index += 1
        
        # If we reached the end of the path, stop the animation
        if self.animation_index >= len(self.animation_path):
            self.animation_running = False
            self.solution_count_label.setText(f"Animation completed. Robot arm at target position.")
            
            # Print the complete path information including angles
            self.print_path_information()
            return
        
        # Schedule the next step if animation is still running
        if self.animation_running:
            # Get animation speed (inversely proportional to delay)
            speed = self.animation_speed_slider.value()
            delay = 1000 // speed  # 100-1000ms
            
            QTimer.singleShot(delay, self.perform_animation_step)
    
    def print_path_information(self):
        """
        Print detailed information about the path trajectory and joint angles.
        Shows all points on the path and the corresponding joint angles needed to reach each point.
        """
        try:
            path_type = self.path_type_combo.currentText()
            print("\n" + "="*80)
            print(f"PATH TRAJECTORY INFORMATION: {path_type}")
            print("="*80)
            
            print(f"Total Points: {len(self.animation_path)}")
            print(f"Configuration: {self.animation_config if self.animation_config else 'Auto (Best)'}")
            
            print("\nSTARTING POINT:")
            start_point = self.animation_path[0]
            print(f"Position (X, Y, Z): ({start_point[0]:.2f}, {start_point[1]:.2f}, {start_point[2]:.2f}) cm")
            
            print("\nTARGET POINT:")
            end_point = self.animation_path[-1]
            print(f"Position (X, Y, Z): ({end_point[0]:.2f}, {end_point[1]:.2f}, {end_point[2]:.2f}) cm")
            
            print("\nDETAILED PATH INFORMATION:")
            print("-"*80)
            print("Point #  |  Position (X, Y, Z) cm  |  Joint Angles (Base, Shoulder, Elbow, Wrist) degrees  |  Config  |  Error")
            print("-"*80)
            
            for i, (point, solution) in enumerate(zip(self.animation_path, self.animation_solutions)):
                x, y, z = point
                
                if solution:
                    base_angle, shoulder_angle, elbow_angle, wrist_angle = solution
                    config = "Elbow-Up" if elbow_angle < 0 else "Elbow-Down"
                    
                    # Calculate error
                    # Save original state
                    original_angles = (self.robot.base_angle, self.robot.shoulder_angle, 
                                      self.robot.elbow_angle, self.robot.wrist_angle)
                    
                    # Set angles to check error
                    self.robot.set_joint_angles(*solution)
                    end_pos = self.robot.end_effector
                    error = np.linalg.norm(np.array([x, y, z]) - end_pos)
                    
                    # Restore original state
                    self.robot.set_joint_angles(*original_angles)
                    
                    # Convert angles to degrees for display
                    base_deg = np.degrees(base_angle)
                    shoulder_deg = np.degrees(shoulder_angle)
                    elbow_deg = np.degrees(elbow_angle)
                    wrist_deg = np.degrees(wrist_angle)
                    
                    print(f"{i+1:3d}  |  ({x:6.2f}, {y:6.2f}, {z:6.2f})  |  ({base_deg:6.2f}, {shoulder_deg:6.2f}, {elbow_deg:6.2f}, {wrist_deg:6.2f})  |  {config:8s}  |  {error:.4f}")
                else:
                    print(f"{i+1:3d}  |  ({x:6.2f}, {y:6.2f}, {z:6.2f})  |  No valid solution found  |  N/A  |  N/A")
            
            print("="*80)
            print("Path information has been printed to the console.")
            
            # Also export to a file
            self.export_path_to_file()
            
        except Exception as e:
            print(f"Error printing path information: {str(e)}")
    
    def export_path_to_file(self):
        """Export the path trajectory and joint angles to a CSV file."""
        try:
            import csv
            filename = f"path_trajectory_{self.path_type_combo.currentText().replace(' ', '_').lower()}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Point #', 'X (cm)', 'Y (cm)', 'Z (cm)', 
                                 'Base Angle (deg)', 'Shoulder Angle (deg)', 'Elbow Angle (deg)', 'Wrist Angle (deg)', 
                                 'Configuration', 'Error (cm)'])
                
                for i, (point, solution) in enumerate(zip(self.animation_path, self.animation_solutions)):
                    x, y, z = point
                    
                    if solution:
                        base_angle, shoulder_angle, elbow_angle, wrist_angle = solution
                        config = "Elbow-Up" if elbow_angle < 0 else "Elbow-Down"
                        
                        # Calculate error
                        self.robot.set_joint_angles(*solution)
                        end_pos = self.robot.end_effector
                        error = np.linalg.norm(np.array([x, y, z]) - end_pos)
                        
                        # Convert angles to degrees
                        base_deg = np.degrees(base_angle)
                        shoulder_deg = np.degrees(shoulder_angle)
                        elbow_deg = np.degrees(elbow_angle)
                        wrist_deg = np.degrees(wrist_angle)
                        
                        writer.writerow([
                            i+1, 
                            f"{x:.2f}", 
                            f"{y:.2f}", 
                            f"{z:.2f}",
                            f"{base_deg:.2f}",
                            f"{shoulder_deg:.2f}",
                            f"{elbow_deg:.2f}",
                            f"{wrist_deg:.2f}",
                            config,
                            f"{error:.4f}"
                        ])
                    else:
                        writer.writerow([
                            i+1, 
                            f"{x:.2f}", 
                            f"{y:.2f}", 
                            f"{z:.2f}",
                            "N/A", "N/A", "N/A", "N/A",
                            "No Solution",
                            "N/A"
                        ])
            
            print(f"Path information has been exported to '{filename}'")
            
        except Exception as e:
            print(f"Error exporting path to file: {str(e)}")
    
    
    def trigger_real_arm_path_execution(self):
        if not self.animation_solutions or not all(s is not None for s in self.animation_solutions):
            QMessageBox.warning(self, "Path Execution Failed", "No valid path generated or path contains unreachable points.")
            return

        reply = QMessageBox.question(self, "Confirm Real Arm Movement",
                                     "Ensure the robot's workspace is clear. "
                                     "Are you sure you want to execute the path on the real robot arm?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.No:
            return

        self.follow_path_real_button.setEnabled(False)
        self.cancel_path_execution_flag = False # Reset flag

        # Setup progress dialog for path execution
        if self.progress_dialog: # Close any old one
            self.progress_dialog.cancel()

        self.progress_dialog = QProgressDialog("Preparing to execute path on real arm...", "Cancel", 0, len(self.animation_solutions), self)
        self.progress_dialog.setWindowTitle("Real Arm Path Execution")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self.cancel_real_arm_path_execution) # Connect cancel signal
        self.progress_dialog.show()
        
        current_gripper_value = self.gripper_slider.value() # Get current gripper state from FK tab
        animation_speed_value = self.animation_speed_slider.value()

        # Create and start the thread
        self._path_execution_worker = PathExecutionWorker(
            self.animation_solutions, # This should be list of [base_rad, shoulder_rad, elbow_rad, wrist_rad]
            current_gripper_value,
            animation_speed_value,
            self # Pass reference to main GUI
        )
        self._path_execution_thread = QThread()
        self._path_execution_worker.moveToThread(self._path_execution_thread)

        # Connect worker signals to GUI slots
        self._path_execution_worker.status_signal.connect(self.update_path_execution_status_slot)
        self._path_execution_worker.progress_signal.connect(self.update_path_execution_progress_slot)
        self._path_execution_worker.finished_signal.connect(self.path_execution_finished_slot)
        
        # Cleanup thread and worker when done
        self._path_execution_worker.finished_signal.connect(self._path_execution_thread.quit)
        self._path_execution_worker.finished_signal.connect(self._path_execution_worker.deleteLater)
        self._path_execution_thread.finished.connect(self._path_execution_thread.deleteLater)
        
        self._path_execution_thread.started.connect(self._path_execution_worker.run)
        self._path_execution_thread.start()

    def cancel_real_arm_path_execution(self):
        print("Real arm path execution cancel requested.")
        self.cancel_path_execution_flag = True # Signal the worker thread to stop
        if self._path_execution_worker:
            self._path_execution_worker.stop() # Tell worker to stop gracefully
        # The progress dialog will be closed by the finished_slot upon cancellation message
        self.update_path_execution_status_slot("Cancelling...")
        # Button re-enabled in path_execution_finished_slot
    
    def export_solutions(self):
        """Export the current solutions to a CSV file"""
        if not self.solutions:
            print("No solutions to export")
            return
        
        try:
            import csv
            with open('robot_arm_solutions.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Solution', 'Base (deg)', 'Shoulder (deg)', 'Elbow (deg)', 
                                'Wrist (deg)', 'Configuration', 'Error (cm)', 
                                'Target X', 'Target Y', 'Target Z'])
                
                for i, solution in enumerate(self.solutions):
                    angles = solution['angles']
                    writer.writerow([
                        i+1,
                        f"{np.degrees(angles[0]):.2f}",
                        f"{np.degrees(angles[1]):.2f}",
                        f"{np.degrees(angles[2]):.2f}",
                        f"{np.degrees(angles[3]):.2f}",
                        solution['config'],
                        f"{solution['error']:.4f}",
                        f"{solution['target'][0]:.2f}",
                        f"{solution['target'][1]:.2f}",
                        f"{solution['target'][2]:.2f}"
                    ])
            
            print("Solutions exported to 'robot_arm_solutions.csv'")
        except Exception as e:
            print(f"Error exporting solutions: {str(e)}")
    
    def visualize_all_solutions(self):
        """Create a static visualization of all current solutions"""
        if not self.solutions:
            print("No solutions to visualize")
            return
        
        try:
            # Create a new figure for the visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Set axis limits
            max_reach = self.robot.base_height + self.robot.shoulder_length + self.robot.elbow_length + self.robot.gripper_length
            ax.set_xlim(-max_reach, max_reach)
            ax.set_ylim(-max_reach, max_reach)
            ax.set_zlim(0, max_reach)
            
            # Set labels and title
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_zlabel('Z (cm)')
            ax.set_title('Multiple Inverse Kinematics Solutions')
            
            # Equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Plot the target point
            target = self.solutions[0]['target']
            ax.scatter([target[0]], [target[1]], [target[2]], color='red', s=100, label='Target')
            
            # Plot each solution with a different color
            colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']
            
            # Limit to 8 solutions for clarity
            solutions_to_plot = min(8, len(self.solutions))
            
            for i in range(solutions_to_plot):
                solution = self.solutions[i]
                base_angle, shoulder_angle, elbow_angle, wrist_angle = solution['angles']
                
                # Remember original angles
                original_angles = [self.robot.base_angle, self.robot.shoulder_angle, 
                                 self.robot.elbow_angle, self.robot.wrist_angle]
                
                # Set the robot to this solution
                self.robot.set_joint_angles(base_angle, shoulder_angle, elbow_angle, wrist_angle)
                
                # Get positions for plotting
                x_points, y_points, z_points = self.robot.get_arm_positions()
                
                # Plot the arm with a unique color
                color = colors[i % len(colors)]
                config = solution['config']
                ax.plot(x_points, y_points, z_points, 'o-', linewidth=2, markersize=6, 
                      color=color, label=f'Solution {i+1} ({config})')
                
                # Restore original angles
                self.robot.set_joint_angles(*original_angles)
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Save and show the visualization
            plt.savefig('all_solutions.png')
            print("Visualization saved as 'all_solutions.png'")
            
            # Try to display the figure (may not work in all environments)
            try:
                plt.show(block=False)
            except:
                pass
            
            # Restore the original plot
            self.update_plot()
            
        except Exception as e:
            print(f"Error visualizing solutions: {str(e)}")
            # Restore the plot
            self.update_plot()
    
    def update_plot(self):
        """Update the 3D plot with the current robot arm position"""
        # Get the arm positions
        x_points, y_points, z_points = self.robot.get_arm_positions()
        
        # Update the arm line
        self.arm_line.set_data(x_points, y_points)
        self.arm_line.set_3d_properties(z_points)
        
        # Update the end effector point
        self.end_effector_point.set_data([self.robot.end_effector[0]], [self.robot.end_effector[1]])
        self.end_effector_point.set_3d_properties([self.robot.end_effector[2]])
        
        # Update the gripper visualization
        left_x, left_y, left_z, right_x, right_y, right_z = self.robot.get_gripper_positions()
        
        self.left_gripper.set_data(left_x, left_y)
        self.left_gripper.set_3d_properties(left_z)
        
        self.right_gripper.set_data(right_x, right_y)
        self.right_gripper.set_3d_properties(right_z)
        
        # Redraw the canvas
        self.canvas.draw()



class PathExecutionWorker(QObject):
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int) # current_step, total_steps
    finished_signal = pyqtSignal(str)    # message

    def __init__(self, solutions_rad, gripper_val_0_100, anim_speed_1_10, gui_instance_ref, parent=None):
        super().__init__(parent)
        self.solutions_rad = solutions_rad # List of [base, shoulder, elbow, wrist] in RADIANS
        self.gripper_val_0_100 = gripper_val_0_100
        self.animation_speed = anim_speed_1_10 # Slider value 1-10
        self.gui = gui_instance_ref # To call send_angles_to_pi and access cancel flag
        self._running = True

    def run(self):
        if not self.solutions_rad or not all(s is not None for s in self.solutions_rad):
            self.finished_signal.emit("Error: Path contains invalid or no solutions.")
            return

        num_points = len(self.solutions_rad)
        self.status_signal.emit(f"Starting path ({num_points} points)...")

        # Calculate delay: slider 1 = max delay, slider 10 = min delay
        # Example: speed 1 maps to ~2s delay, speed 10 maps to ~0.2s delay.
        # You can adjust these min/max delays.
        min_step_delay_s = 0.1  # Minimum time per step (for fastest speed)
        max_step_delay_s = 0.2  # Maximum time per step (for slowest speed)
        
        # Linear interpolation for delay based on slider value (1-10)
        if self.animation_speed == 1:
            step_delay_s = max_step_delay_s
        elif self.animation_speed == 10:
            step_delay_s = min_step_delay_s
        else:
            step_delay_s = max_step_delay_s - ((self.animation_speed - 1) / (10 - 1)) * (max_step_delay_s - min_step_delay_s)
        
        print(f"Real arm path execution: {num_points} points, Step Delay: {step_delay_s:.2f}s (Slider: {self.animation_speed})")

        for i, angles_rad_tuple in enumerate(self.solutions_rad):
            if not self._running or self.gui.cancel_path_execution_flag:
                self.finished_signal.emit("Path execution cancelled by user.")
                return

            base_r, shoulder_r, elbow_r, wrist_r = angles_rad_tuple

            angles_to_send = {
                "base": int(np.degrees(base_r)),
                "shoulder": int(np.degrees(shoulder_r)),
                "elbow": int(np.degrees(elbow_r)),
                "wrist": int(np.degrees(wrist_r)),
                "gripper_wrist": int(np.interp(self.gripper_val_0_100, [0, 100], [0, 90]))
            }
            
            # Send angles to Pi (this function is already designed to be non-blocking for the GUI)
            self.gui.send_angles_to_pi(angles_to_send)
            
            # Update simulator in GUI (optional, but good for visual feedback)
            # This needs to be thread-safe if it modifies GUI elements directly.
            # A safer way is to emit a signal for the GUI to update its robot model.
            # For now, let's assume the main GUI timer will update the plot based on self.robot changes.
            # To make the GUI plot update immediately to reflect this step:
            self.gui.robot.set_joint_angles(base_r, shoulder_r, elbow_r, wrist_r, self.gripper_val_0_100 / 100.0)
            # QMetaObject.invokeMethod(self.gui, "update_plot", Qt.QueuedConnection) # If direct update needed

            self.status_signal.emit(f"Executing point {i + 1}/{num_points}...")
            self.progress_signal.emit(i + 1, num_points)
            
            time.sleep(step_delay_s) # This sleep is crucial for the real arm movement

        self.finished_signal.emit("Path execution completed successfully.")

    def stop(self):
        self._running = False
        
        
class VideoThread(QThread):
    """
    Ayrı bir thread'de Raspberry Pi'den video akışını alır.
    """
    change_pixmap_signal = pyqtSignal(QImage)
    connection_status_signal = pyqtSignal(str) # Bağlantı/hata durumu için sinyal

    def __init__(self, stream_url, parent=None): # parent eklendi
        super().__init__(parent) # parent ile çağrı
        self._run_flag = True
        self.stream_url = stream_url
        self.cap = None
        self.setObjectName("VideoAlimThread") # Thread'e isim vermek debug için faydalı

    def run(self):
        print(f"[{self.objectName()}] Video akışı başlatılıyor: {self.stream_url}")
        self.connection_status_signal.emit(f"Bağlanılıyor: {self.stream_url}")
        
        self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG) # FFMPEG backend'i denenebilir
        
        if not self.cap.isOpened():
            error_msg = f"Hata: Video akışına bağlanılamadı!\nURL: {self.stream_url}\nPi sunucusunun çalıştığından ve kameranın aktif olduğundan emin olun."
            self.connection_status_signal.emit(error_msg)
            print(f"[{self.objectName()}] {error_msg}")
            return # Thread'i sonlandır

        self.connection_status_signal.emit("Successfully connected to video stream.")
        print(f"[{self.objectName()}] Successfully connected to video stream.")
        
        while self._run_flag:
            if not self.cap.isOpened(): # Her döngüde kontrol et
                self.connection_status_signal.emit("Hata: Video bağlantısı koptu. Yeniden deneniyor...")
                print(f"[{self.objectName()}] Video bağlantısı koptu. Yeniden deneniyor...")
                time.sleep(1) # Yeniden denemeden önce bekle
                self.cap.release()
                self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    self.connection_status_signal.emit("Hata: Yeniden bağlanma başarısız.")
                    print(f"[{self.objectName()}] Yeniden bağlanma başarısız.")
                    break # Döngüden çık
                else:
                    self.connection_status_signal.emit("Video akışına yeniden bağlanıldı.")
                    print(f"[{self.objectName()}] Video akışına yeniden bağlanıldı.")

            ret, cv_img = self.cap.read()
            if ret:
                try:
                    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(qt_image.copy())
                except Exception as e:
                    print(f"[{self.objectName()}] Video karesi işleme hatası: {e}")
            else:
                # Kare okunamadıysa kısa bir süre bekle, bağlantı kopmuş olabilir
                # print(f"[{self.objectName()}] Akıştan kare okunamadı (ret=False).") # Çok sık log basabilir
                time.sleep(0.05) # CPU'yu yormamak için küçük bir bekleme
            
            # Thread'in olayları işlemesine izin ver (özellikle _run_flag kontrolü için)
            # QThread.msleep(10) # veya time.sleep(0.01)

        if self.cap:
            self.cap.release()
        print(f"[{self.objectName()}] Video thread finished.")
        # connection_status_signal burada da emit edilebilir, ancak finished sinyali daha uygun
        # self.connection_status_signal.emit("Video akışı durduruldu (run metodu bitti).")


    def stop(self):
        """Stop Live Stream and the thread."""
        print(f"[{self.objectName()}] Stop signal is received.")
        self._run_flag = False
        # self.cap.release() # run() içinde zaten yapılıyor, burada da çağrılabilir ama dikkatli olunmalı
        # self.wait(2000) # Thread'in sonlanması için bekleme, finished sinyali beklenmeli
        # QThread'in kendi sonlanma mekanizmasını kullanmak daha iyi.
        # GUI'den `thread.quit()` ve `thread.wait()` çağrılabilir.


class GeminiVisionThread(QThread):
    """
    Ayrı bir thread'de Gemini API'ye görüntü gönderir ve cevabı alır.
    """
    gemini_response_signal = pyqtSignal(str)
    gemini_error_signal = pyqtSignal(str)

    def __init__(self, api_key, image_bytes, prompt, parent=None):
        super().__init__(parent)
        self.api_key = api_key
        self.image_bytes = image_bytes
        self.prompt = prompt
        self.setObjectName("GeminiAnalizThread")

    def run(self):
        if not genai:
            self.gemini_error_signal.emit("Google Generative AI library is not installed.")
            return
        try:
            print(f"[{self.objectName()}] Gemini API connecting...")
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')

            image_part = {
                "mime_type": "image/jpeg",
                "data": self.image_bytes
            }
            print(f"[{self.objectName()}] Prompt: '{self.prompt}', Image Size: {len(self.image_bytes)} bytes")
            response = model.generate_content([self.prompt, image_part])
            print(f"[{self.objectName()}] Gemini API gives result.")
            self.gemini_response_signal.emit(response.text)

        except Exception as e:
            error_message = f"Gemini API hatası: {str(e)}"
            print(f"[{self.objectName()}] {error_message}")
            if "API_KEY_INVALID" in str(e) or "API_KEY_MISSING" in str(e) :
                 error_message += "\nLütfen GEMINI_API_KEY ortam değişkenini kontrol edin."
            self.gemini_error_signal.emit(error_message)
        print(f"[{self.objectName()}] Gemini thread sonlandırıldı.")

            
            
def main():
    app = QApplication(sys.argv)
    window = ImprovedRobotArmGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


