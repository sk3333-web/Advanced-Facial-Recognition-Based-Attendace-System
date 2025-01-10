import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import pickle
from collections import defaultdict
import logging
from ultralytics import YOLO
import threading
import torch
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console logging
        # Uncomment the following line to enable file logging
        # logging.FileHandler("attendance_system.log"),
    ]
)

# Constants
BASE_PATH = r" "  # Directory with employee subdirectories
CSV_DIRECTORY = r" "  # Directory for CSV files
ENCODINGS_FILE = os.path.join(CSV_DIRECTORY, "encodings.pkl")
ATTENDANCE_FILE = os.path.join(CSV_DIRECTORY, 'Attendance.csv')
INTERVAL_ATTENDANCE_FILE = os.path.join(CSV_DIRECTORY, 'IntervalAttendance.csv')
DISTANCE_THRESHOLD = 0.95  # Adjusted for better recognition
YOLO_MODEL_PATH = r"yolov8n-face.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 3  # Process every 3rd frame
INTERVAL_DURATION = timedelta(minutes=5)  # 5-minute intervals

# RTSP streams
RTSP_STREAMS = [
    'multiple rtsp links
]

# Initialize variables
employee_encodings = {}
classNames = []
attendance_frame_count = defaultdict(int)
last_attendance = defaultdict(lambda: datetime.min)
interval_presence = set()
interval_start_time = datetime.now()

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run the script in headless mode without GUI display.'
    )
    return parser.parse_args()

args = parse_arguments()
headless_mode = args.headless
logging.info(f"Headless mode activated: {headless_mode}")

# Ensure CSV directory exists
if not os.path.exists(CSV_DIRECTORY):
    try:
        os.makedirs(CSV_DIRECTORY)
        logging.info(f"Created CSV directory at {CSV_DIRECTORY}")
    except Exception as e:
        logging.error(f"Failed to create CSV directory at {CSV_DIRECTORY}: {e}")
        exit(1)  # Exit the script if the directory cannot be created

# Load YOLOv8n face detection model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")
try:
    yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO model from {YOLO_MODEL_PATH}: {e}")
    exit(1)  # Exit if the model cannot be loaded

def generate_encodings():
    """
    Traverse the BASE_PATH directory, process each employee's images,
    compute face encodings, and populate employee_encodings and classNames.
    """
    global employee_encodings, classNames
    classNames = []
    employee_encodings = {}

    # Iterate through each employee's directory
    for employee_name in os.listdir(BASE_PATH):
        employee_dir = os.path.join(BASE_PATH, employee_name)
        if not os.path.isdir(employee_dir):
            continue  # Skip files, only process directories

        classNames.append(employee_name)
        encodings = []

        # Iterate through each image in the employee's directory
        for img_name in os.listdir(employee_dir):
            img_path = os.path.join(employee_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Unable to read image: {img_path}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(img_rgb)
                if not face_locations:
                    logging.warning(f"No face found in image: {img_path}")
                    continue
                face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
                encodings.extend(face_encodings)
            except Exception as e:
                logging.error(f"Error processing image {img_path}: {e}")

        if encodings:
            employee_encodings[employee_name] = {'encodings': encodings}
            logging.info(f"Generated {len(encodings)} encodings for {employee_name}")
        else:
            logging.warning(f"No encodings found for {employee_name}")

    # Save the encodings to the file
    try:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump((employee_encodings, classNames), f)
        logging.info(f"Encodings generated and saved to {ENCODINGS_FILE}")
    except Exception as e:
        logging.error(f"Failed to save encodings to {ENCODINGS_FILE}: {e}")

def load_encodings():
    global employee_encodings, classNames
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                employee_encodings, classNames = pickle.load(f)
            logging.info('Encodings Loaded Successfully')
        except Exception as e:
            logging.error(f'Error loading encodings: {e}')
            employee_encodings = {}
            classNames = []
    else:
        logging.info('No existing encodings found. Generating new encodings.')
        generate_encodings()

def preprocess_face(face_image):
    face_image = cv2.resize(face_image, (112, 112))
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    return face_image_rgb

def mark_attendance(name, camera_source):
    """
    Log overall attendance to Attendance.csv if not already marked within the cooldown period (8 hours).
    """
    name = name.upper()  # Normalize name to uppercase
    now = datetime.now()
    cooldown_period = timedelta(hours=8)  # Cooldown period of 8 hours

    if now - last_attendance[name] >= cooldown_period:  # Check if cooldown period has passed
        try:
            with open(ATTENDANCE_FILE, 'a+', encoding='utf-8') as f:
                dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{name},{dt_string},{camera_source}\n")
            logging.info(f"Attendance marked for {name} at {dt_string} from Camera {camera_source}")
            last_attendance[name] = now
        except Exception as e:
            logging.error(f"Failed to write to {ATTENDANCE_FILE}: {e}")
    else:
        logging.info(f"Attendance for {name} not marked. Cooldown period not yet over.")

def mark_interval_attendance(camera_source):
    """
    At the end of each interval, mark employees as present or absent and log them to IntervalAttendance.csv.
    """
    global interval_presence, interval_start_time
    now = datetime.now()

    # List all employees and normalize case
    all_employees = {name.upper() for name in classNames}

    # Determine absent employees
    present_employees = {name.upper() for name in interval_presence}
    absent_employees = all_employees - present_employees

    # Log interval attendance
    try:
        with open(INTERVAL_ATTENDANCE_FILE, 'a+', encoding='utf-8') as f:
            for emp in present_employees:
                f.write(f"{emp},Present,{interval_start_time.strftime('%Y-%m-%d %H:%M:%S')},{now.strftime('%Y-%m-%d %H:%M:%S')},{camera_source}\n")
            for emp in absent_employees:
                f.write(f"{emp},Absent,{interval_start_time.strftime('%Y-%m-%d %H:%M:%S')},{now.strftime('%Y-%m-%d %H:%M:%S')},{camera_source}\n")
        logging.info(f"Interval attendance recorded. Present: {present_employees}, Absent: {absent_employees} from Camera {camera_source}")
    except Exception as e:
        logging.error(f"Failed to write to {INTERVAL_ATTENDANCE_FILE}: {e}")

    # Reset interval tracking
    interval_presence.clear()
    interval_start_time = now

def initialize_csv_files():
    """
    Initialize the Attendance.csv and IntervalAttendance.csv files with headers if they do not exist.
    """
    try:
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
                f.write("Employee,Timestamp,Camera Source\n")
            logging.info(f"Initialized {ATTENDANCE_FILE} with headers.")

        if not os.path.exists(INTERVAL_ATTENDANCE_FILE):
            with open(INTERVAL_ATTENDANCE_FILE, 'w', encoding='utf-8') as f:
                f.write("Employee,Status,Start Time,End Time,Camera Source\n")
            logging.info(f"Initialized {INTERVAL_ATTENDANCE_FILE} with headers.")
    except Exception as e:
        logging.error(f"Failed to initialize CSV files: {e}")

# Add verification threshold
verification_threshold = 10  # Number of consecutive frames required for consistent labeling
consecutive_label_count = defaultdict(int)  # Tracks consecutive detections for each label

def process_frame(img, camera_source, headless_mode):
    """
    Process each frame for face detection and attendance marking with verification logic.
    """
    global interval_presence, interval_start_time

    # Check interval duration
    if datetime.now() - interval_start_time >= INTERVAL_DURATION:
        mark_interval_attendance(camera_source)

    # YOLO for face detection
    try:
        results = yolo_model(img, conf=YOLO_CONFIDENCE_THRESHOLD)
    except Exception as e:
        logging.error(f"YOLO model failed to process image from Camera {camera_source}: {e}")
        return img  # Return original image if YOLO fails

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            face_image = img[y1:y2, x1:x2]
            if face_image.size == 0:
                logging.warning(f"Empty face image detected from Camera {camera_source}. Skipping.")
                continue
            face_image = preprocess_face(face_image)
            encodeFace = face_recognition.face_encodings(face_image)
            if encodeFace:
                name = 'Unknown'
                min_distance = float("inf")

                # Find the closest match
                for employee, data in employee_encodings.items():
                    for stored_encoding in data['encodings']:
                        distance = np.linalg.norm(encodeFace[0] - stored_encoding)
                        if distance < min_distance:
                            min_distance = distance
                            name = employee.upper()

                if min_distance < DISTANCE_THRESHOLD:
                    # Increment count for consecutive detection
                    consecutive_label_count[name] += 1
                    if consecutive_label_count[name] >= verification_threshold:
                        # Mark attendance and reset counter after verification
                        if name not in interval_presence:
                            interval_presence.add(name)  # Track interval presence
                        mark_attendance(name, camera_source)  # Mark overall attendance
                        consecutive_label_count[name] = 0
                        color = (0, 255, 0)  # Green for verified faces
                    else:
                        color = (255, 255, 0)  # Yellow for pending verification
                else:
                    # Reset count for unverified name
                    consecutive_label_count[name] = 0
                    color = (0, 0, 255)  # Red for unknown faces
            else:
                name = 'Unknown'
                color = (0, 0, 255)  # Red for unknown faces

            # Draw bounding box and label if not headless
            if not headless_mode:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    return img

def process_stream(stream_url, window_name, headless=False):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        logging.error(f"Cannot open video stream: {stream_url}")
        return

    camera_source = stream_url.split('/')[-1].split(':')[0]  # Extract camera identifier (e.g., '301')

    frame_count = 0
    try:
        while True:
            success, img = cap.read()
            if not success:
                logging.info(f"Failed to read frame from {stream_url}.")
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            processed_img = process_frame(img, camera_source, headless_mode)

            if not headless:
                cv2.imshow(window_name, processed_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info(f"Exit key pressed for {window_name}. Shutting down.")
                    break
    except KeyboardInterrupt:
        logging.info(f"KeyboardInterrupt received. Shutting down {window_name}.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyWindow(window_name)

def main():
    load_encodings()
    initialize_csv_files()
    threads = []
    for i, stream_url in enumerate(RTSP_STREAMS):
        window_name = f"Stream {i+1}"
        thread = threading.Thread(target=process_stream, args=(stream_url, window_name, headless_mode))
        threads.append(thread)
        thread.start()

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received in main thread. Attempting to shut down gracefully.")
        # Threads will handle their own shutdown
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    main()
