# Face Recognition Attendance System

## Overview

The **Face Recognition Attendance System** is an innovative solution designed to automate attendance tracking in organizations. By leveraging state-of-the-art computer vision techniques, the system uses YOLO for real-time face detection and the `face_recognition` library for accurate face matching. This robust system processes video streams from multiple cameras, marking attendance and logging interval-based presence effortlessly.

### Key Benefits
- Reduces manual errors in attendance tracking.
- Saves time and enhances operational efficiency.
- Scalable for multi-camera setups.
- Provides detailed insights through interval-based logging.

## Features

- **Real-Time Face Recognition**: Detects and recognizes employee faces from live RTSP streams.
- **Attendance Logging**: Maintains detailed logs with timestamps and camera identifiers.
- **Interval-Based Tracking**: Tracks employee presence within configurable time intervals (default: 5 minutes).
- **Multi-Camera Support**: Handles multiple video streams simultaneously for large-scale deployments.
- **Customizable Thresholds**: Ensures recognition accuracy with adjustable verification and distance thresholds.
- **Headless Mode**: Option to run the system without a graphical user interface for server-based setups.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (for optimal performance).

### Software
- Python 3.8 or higher
- Required Libraries:
  - `opencv-python`
  - `numpy`
  - `face_recognition`
  - `torch`
  - `ultralytics`

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<username>/face-recognition-attendance-system.git
cd face-recognition-attendance-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare YOLOv8 Weights
Download the YOLOv8 face detection model weight file (`yolov8n-face.pt`) and place it in the project directory.

### 4. Set Up Directory Structure

- **Employee Images**:
  - Create a directory containing subfolders for each employee, where each subfolder includes the employee's images.
  ```
  BASE_PATH = "<path-to-employee-images>"
  ```
- **CSV Logs**:
  - Create a directory for storing attendance logs.
  ```
  CSV_DIRECTORY = "<path-to-csv-directory>"
  ```

## Usage

### 1. Generate Face Encodings
Generate encodings for all employee images stored in the `BASE_PATH` directory. This step is necessary for the system to recognize faces during runtime.
```bash
python Attendance_system.py --generate_encodings
```

### 2. Run the Attendance System
Start real-time face recognition and attendance tracking.
```bash
python Attendance_system.py
```

#### Command-Line Arguments
- `--headless`: Runs the system in a server-friendly, GUI-less mode.

Example:
```bash
python Attendance_system.py --headless
```

### 3. Review Attendance Logs
Logs are automatically saved in the specified `CSV_DIRECTORY`:
- **Overall Attendance**:
  - File: `Attendance.csv`
  - Columns: Employee Name, Timestamp, Camera Source
- **Interval-Based Attendance**:
  - File: `IntervalAttendance.csv`
  - Columns: Employee Name, Status (Present/Absent), Start Time, End Time, Camera Source

### 4. Customize Parameters
- **Face Verification Threshold**:
  Adjust the `DISTANCE_THRESHOLD` in the script to fine-tune face recognition accuracy.
- **Frame Skipping**:
  Modify the `FRAME_SKIP` value to process every nth frame for performance optimization.

## Project Structure

```
├── Attendance_system.py          # Main script for face recognition attendance
├── requirements.txt              # List of required dependencies
├── yolov8n-face.pt               # YOLOv8 face detection model weights
├── employee_images/              # Directory containing employee images
├── csv_logs/                     # Directory for CSV logs
    ├── Attendance.csv            # Overall attendance log
    ├── IntervalAttendance.csv    # Interval-based attendance log
```

## Screenshots

### Live Video Feed with Face Recognition
![Live Face Recognition](image.png)

### Attendance Logs
- Example of `Attendance.csv`:
  ```csv
  Employee,Timestamp,Camera Source
  JOHN DOE,2025-01-10 09:30:45,Camera 1
  ```

## Future Enhancements

- **Mobile App Integration**: Enable employees to view attendance records through a dedicated mobile app.
- **Automated Notifications**: Notify employees of attendance updates via email or messaging platforms.
- **Edge Deployment**: Optimize the system for edge devices like NVIDIA Jetson Nano.
- **Advanced Analytics**: Generate heatmaps and detailed reports on employee activity.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to contribute to this project by submitting issues or pull requests on GitHub!
