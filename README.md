# Autonomous and Teleoperated 4-DoF Robot Arm with Chassis

A comprehensive robotic system featuring a 4 degrees-of-freedom robot arm with mobile chassis, combining simulation, autonomous control, teleoperation, and computer vision capabilities.

![Robot Arm Demo](docs/images/robot_demo.gif)

## 🎬 Project Demonstrations

### Quick Overview (1 minute)
[![1-Minute Demo](https://img.youtube.com/vi/iX-ZMjMz9j0/0.jpg)](https://youtu.be/iX-ZMjMz9j0?si=5JM2QOXZX02kEiOU)

**[🎥 Watch 1-Minute Project Overview](https://youtu.be/iX-ZMjMz9j0?si=5JM2QOXZX02kEiOU)**

A concise demonstration showcasing the key features and capabilities of the robotic system.

### Technical Deep Dive (6 minutes)
[![6-Minute Technical Demo](https://img.youtube.com/vi/yufLHJdyqQU/0.jpg)](https://youtu.be/yufLHJdyqQU?si=9JoUdrnNEJq3PlhQ)

**[🎥 Watch 6-Minute Technical Presentation](https://youtu.be/yufLHJdyqQU?si=9JoUdrnNEJq3PlhQ)**

Detailed technical presentation covering:
- System architecture and components
- Kinematics implementation and path planning
- Computer vision and AI integration
- Real-time control and simulation comparison
- Hardware setup and software features

## 🎯 Project Overview

This project implements a complete robotic ecosystem featuring:
- **4-DoF Robot Arm**: Precise manipulation with base, shoulder, elbow, and wrist joints
- **Mobile Chassis**: 4-wheel mecanum drive system for omnidirectional movement
- **Dual Control Modes**: Advanced simulation environment and real hardware control
- **Computer Vision**: Real-time object detection and tracking using Pi Camera
- **AI Integration**: Gemini Vision API for intelligent scene analysis
- **Path Planning**: Multiple trajectory types with collision avoidance
- **Real-time Streaming**: Live video feed with computer vision overlay

## ✨ Key Features

### Simulation & Control
- **Forward Kinematics**: Real-time joint control with visual feedback
- **Inverse Kinematics**: Target position solving with multiple configuration options
- **Path Planning**: Straight line, circular arcs, and figure-8 trajectories
- **Multiple Solutions**: Automatic elbow-up/elbow-down configuration detection
- **Real-time Visualization**: Interactive 3D plotting with matplotlib

### Hardware Integration
- **Servo Control**: Precise 4-DoF arm control via GPIO
- **Chassis Control**: Omnidirectional movement with speed control
- **Camera Integration**: Pi Camera v2 with real-time streaming
- **Object Detection**: Red cube tracking with pose estimation
- **Network Communication**: RESTful API for remote control

### Advanced Features
- **Comparison Mode**: Side-by-side ideal vs. real robot simulation
- **Vision Analysis**: AI-powered scene understanding with Gemini
- **Path Execution**: Automated trajectory following on real hardware
- **Error Analysis**: Position accuracy and joint angle comparison
- **Export Capabilities**: CSV data export and trajectory logging

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Control Computer (PC)                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────┐ │
│  │   GUI Control   │────│  Path Planning  │────│ Vision  │ │
│  │ (raspiw_chassis)│    │ (path_planner)  │    │Analysis │ │
│  └─────────────────┘    └─────────────────┘    └─────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │ WiFi/Ethernet (192.168.145.251:5000)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Raspberry Pi 4B (Hardware Controller)          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────┐ │
│  │  Flask Server   │────│  Servo Control  │────│ Camera  │ │
│  │  (product.py)   │    │   (pigpio)      │    │ Module  │ │
│  └─────────────────┘    └─────────────────┘    └─────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │ GPIO/I2C/SPI
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Physical Hardware                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────┐ │
│  │   Servo Motors  │    │  Chassis Motors │    │ Sensors │ │
│  │  (4-DoF Arm)    │    │ (4WD Platform)  │    │& Camera │ │
│  └─────────────────┘    └─────────────────┘    └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Hardware Requirements

### Essential Components

#### Raspberry Pi 4B Setup
- **Raspberry Pi 4B** (4GB+ recommended)
- **MicroSD Card** (32GB+ Class 10)
- **Pi Camera Module v2** (8MP) or compatible USB webcam
- **GPIO Breakout Board** for easy connections

#### Robot Arm Components
- **4x High-torque Servo Motors** (MG996R, 9.4 kg·cm torque)
  - Base rotation servo (Pin 17)
  - Shoulder servo (Pin 18)
  - Elbow servo (Pin 19)
  - Wrist servo (Pin 12)
- **1x Mini Servo Motor** (SG90, 1.98 kg·cm torque)
  - Gripper servo (Pin 13)

#### Chassis Components
- **4x DC Brushless Motors** (12V, 200 RPM)
- **4x Mecanum Wheels** (80mm diameter, rubber)
- **2x L298N Motor Driver Boards**
- **Chassis Platform** (6mm plexiglass, laser-cut)

#### Power & Electronics
- **14.8V Li-Po Battery** (5000mAh)
- **4x Voltage Regulators** (Step-down: 5V, 7V, 12V)
- **Raspberry Pi Heat Sink Set** (Aluminum, 4-in-1)
- **Metal Servo Mounting Parts** (4x aluminum brackets)
- **Breadboard/PCB** for connections
- **Jumper Wires** and XT60 charging cable

### Pin Configuration

```python
# Servo Pins (BCM numbering)
BASE_PIN = 17
SHOULDER_PIN = 18
ELBOW_PIN = 19
WRIST_PIN = 12
GRIPPER_PIN = 13

# Chassis Motor Pins
FL_MOTOR_PHASE = 5    # Front Left Phase
FL_MOTOR_ENABLE = 6   # Front Left Enable
FR_MOTOR_PHASE = 26   # Front Right Phase
FR_MOTOR_ENABLE = 16  # Front Right Enable
RL_MOTOR_PHASE = 20   # Rear Left Phase
RL_MOTOR_ENABLE = 21  # Rear Left Enable
RR_MOTOR_PHASE = 23   # Rear Right Phase
RR_MOTOR_ENABLE = 24  # Rear Right Enable
```

## 💻 Software Requirements

### Raspberry Pi Dependencies
```bash
# System packages
sudo apt update
sudo apt install python3-pip python3-opencv git

# Python packages
pip3 install flask gpiozero pigpio opencv-python numpy picamera2 requests

# Enable pigpio daemon
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

### Control Computer Dependencies
```bash
# Python packages for GUI and simulation
pip install PyQt5 matplotlib numpy requests opencv-python
pip install google-generativeai  # For Gemini Vision API
```

## 🚀 Installation & Setup

### 1. Raspberry Pi Setup

1. **Flash Raspberry Pi OS** to SD card
2. **Enable SSH, Camera, and GPIO**:
   ```bash
   sudo raspi-config
   # Enable SSH, Camera, GPIO, and I2C
   ```

3. **Clone the repository**:
   ```bash
   git clone https://github.com/mehmetkantar/Autonomous-and-Teleoperated-4-DoF-Robot-Arm-with-chassis.git
   cd Autonomous-and-Teleoperated-4-DoF-Robot-Arm-with-chassis
   ```

4. **Install dependencies**:
   ```bash
   pip3 install -r requirements_pi.txt
   ```

5. **Camera calibration** (required for vision):
   ```bash
   python3 camera_calibration.py
   # Follow on-screen instructions with calibration pattern
   ```

6. **Start the hardware server**:
   ```bash
   python3 product.py --preview  # Add --preview to see detection window
   ```

### 2. Control Computer Setup

1. **Install dependencies**:
   ```bash
   pip install PyQt5 matplotlib numpy requests opencv-python google-generativeai
   ```

2. **Configure network connection**:
   - Update `RASPBERRY_PI_IP` in `raspiw_chassis.py`
   - Default: `192.168.145.251:5000`

3. **Set up Gemini API** (optional):
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

4. **Launch the control interface**:
   ```bash
   python raspiw_chassis.py
   ```

## 🎮 Usage Guide

### Forward Kinematics Mode
- Use sliders to control individual joint angles
- Real-time position feedback and visualization
- Direct servo control with instant hardware response
- Calibration offsets for precise positioning

### Inverse Kinematics Mode
- Input target X, Y, Z coordinates
- Choose arm configuration (elbow up/down/auto)
- Solve for optimal joint angles automatically
- Apply solutions to both simulation and hardware

### Path Planning & Animation
1. **Generate Paths**: Select from multiple trajectory types
   - Straight line
   - Half circle (up/down)
   - Figure-8 patterns
2. **Preview Animation**: Visualize path in simulation
3. **Execute on Hardware**: Transfer path to real robot
4. **Monitor Progress**: Real-time execution status

### Computer Vision Features
- **Live Video Stream**: Real-time camera feed
- **Object Detection**: Red cube tracking and pose estimation
- **AI Analysis**: Gemini Vision for scene understanding
- **Coordinate Mapping**: Automatic target position detection

### Chassis Control
- **Omnidirectional Movement**: Forward, backward, left, right
- **Rotation Control**: In-place turning
- **Speed Adjustment**: Variable speed control
- **Real-time Status**: Movement feedback and monitoring

## 📁 Project Structure

```
├── raspiw_chassis.py           # Main GUI control application
├── robot_arm_improved.py       # Robot arm kinematics and control
├── path_planner.py             # Trajectory generation algorithms
├── comparison_simulation.py    # Ideal vs real robot comparison
├── product.py                  # Raspberry Pi hardware server
├── config/
│   ├── camera_calibration.py         # Servo calibration parameters
│   └── camera_params.pkl       # Camera calibration data
└── docs/
    └── Final Report.pdf # Camera calibration pattern

```

## 🔧 Configuration

### Robot Dimensions (Validated Through FEA)
```python
# In robot_arm_improved.py - validated through structural analysis
self.base_height = 13.0      # cm
self.shoulder_length = 15.0  # cm  
self.elbow_length = 15.0     # cm
self.gripper_length = 4.0    # cm

# Material Properties (PLA - 3D Printed Components)
# Elasticity Modulus = 3 GPa
# Yield Strength = 80 MPa  
# Density = 1.25 g/cm³
```

### Servo Calibration
```python
# In product.py - adjust for your hardware
SERVO_MIN_ANGLE = -90
SERVO_MAX_ANGLE = 90
SERVO_MIN_PW = 0.0005  # Minimum pulse width
SERVO_MAX_PW = 0.0025  # Maximum pulse width
GRIPPER_MIN_ANGLE = 0
GRIPPER_MAX_ANGLE = 90
```

### Network Configuration
```python
# In raspiw_chassis.py - update IP address
RASPBERRY_PI_IP = "192.168.145.251"
RASPBERRY_PI_PORT = 5000
```

## 🧪 Testing & Validation

### Network Tests
```bash
# Test Pi connectivity
curl http://192.168.145.251:5000/test

# Test servo control
curl -X POST http://192.168.145.251:5000/set_angles \
  -H "Content-Type: application/json" \
  -d '{"base": 45, "shoulder": 30, "elbow": -45, "wrist": 0}'
```

## 📈 Performance Specifications

### Mechanical Performance
- **Payload Capacity**: 500g at full extension
- **Repeatability**: ±2mm positioning accuracy
- **Working Envelope**: 45cm radius hemisphere
- **Joint Ranges**: ±90° for all joints
- **Speed**: Variable 0.1-2.0 rad/s per joint

### Control Performance
- **Update Rate**: 100Hz simulation, 10Hz hardware
- **Response Time**: <100ms for servo commands
- **Path Accuracy**: <5mm deviation from planned trajectory
- **Vision Frame Rate**: 30 FPS @ 1296x972 resolution

### System Performance
- **Network Latency**: <50ms on local network
- **Battery Life**: 2-4 hours continuous operation
- **Processing Load**: <30% CPU on Pi 4B
- **Memory Usage**: <512MB RAM

## 🚧 Troubleshooting

### Common Issues

#### Servo Not Responding
```bash
# Check pigpio daemon
sudo systemctl status pigpiod

# Restart if needed
sudo systemctl restart pigpiod

# Verify GPIO permissions
sudo usermod -a -G gpio $USER
```

#### Network Connection Issues
```python
# Test Pi server status
import requests
response = requests.get("http://192.168.145.251:5000/test")
print(response.json())
```

#### Camera Problems
```bash
# Check camera detection
vcgencmd get_camera

# Test camera capture
libcamera-still -o test.jpg
```

#### Inverse Kinematics Fails
- Check target position is within reach (45cm max)
- Verify joint limits are not exceeded
- Try different arm configurations
- Increase error threshold for difficult positions

- **Mehmet Kantar** - *Project Lead* - [@mehmetkantar](https://github.com/mehmetkantar)
- **Zeynep Güvenç** - [@zeynepguvenc](https://github.com/zeynepguvenc)
- **Cemal Efe Gayir** 


*This project demonstrates the integration of advanced robotics concepts including kinematics, path planning, computer vision, and distributed control systems, in an accessible and educational format.*
