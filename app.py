# ----------------------------------
# 📦 IMPORT REQUIRED MODULES
# ----------------------------------
from flask import Flask, request, jsonify, send_from_directory  # Flask essentials: web app, file handling, JSON responses
from werkzeug.utils import secure_filename  # Ensures uploaded filenames are safe (e.g., removes slashes)
import os  # Interact with the OS (for file paths, folders)
import cv2  # OpenCV for image processing
from datetime import datetime  # Used for timestamping
from ultralytics import YOLO  # YOLO model from Ultralytics (used for object detection)
import logging  # For logging events
from flask_cors import CORS  # Enables Cross-Origin Resource Sharing (useful for frontend-backend integration)
import firebase_admin  # Firebase Admin SDK
from firebase_admin import credentials, db  # Firebase auth and real-time database
import numpy as np  # NumPy for calculations like IoU

# ----------------------------------
# 🔍 CONFIGURE LOGGING
# ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)  # Setup structured logging for debugging and tracing

# ----------------------------------
# 🚀 INITIALIZE FLASK APP
# ----------------------------------
app = Flask(__name__)  # Create a Flask app instance
CORS(app)  # Allow requests from other domains (e.g., frontend running on different port)

# ----------------------------------
# 🛠️ CONFIGURATION SETTINGS
# ----------------------------------
UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded images
PROCESSED_FOLDER = 'processed'  # Folder to store processed (annotated) images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}  # Accepted file types
MODEL_PATH = 'model/best.onnx'  # Path to the YOLO ONNX model
FIREBASE_CREDENTIALS = 'serviceAccountKey.json'  # Path to Firebase credentials file
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence score to consider a detection as valid

# ----------------------------------
# ☁️ INITIALIZE FIREBASE
# ----------------------------------
if not os.path.exists(FIREBASE_CREDENTIALS):
    logging.error("Firebase credentials file missing!")
    raise FileNotFoundError("Firebase credentials file not found.")

cred = credentials.Certificate(FIREBASE_CREDENTIALS)  # Load credentials
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ppe-detection-cloud-default-rtdb.firebaseio.com/'  # Connect to Firebase Realtime DB
})

# ----------------------------------
# 📁 ENSURE REQUIRED FOLDERS EXIST
# ----------------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if it doesn't exist
os.makedirs(PROCESSED_FOLDER, exist_ok=True)  # Create processed folder if it doesn't exist

# ----------------------------------
# 🤖 LOAD YOLO MODEL
# ----------------------------------
try:
    model = YOLO(MODEL_PATH)  # Load the YOLOv8 ONNX model
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise  # Stop the app if model fails to load

# ----------------------------------
# 🔎 VALIDATE ALLOWED FILE TYPES
# ----------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS  # Check extension

# ----------------------------------
# 🔢 IoU (Intersection over Union) CALCULATION
# ----------------------------------
def iou(boxA, boxB):
    # Calculate overlap area
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate area of both boxes
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    # Return IoU score
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

# ----------------------------------
# 🚨 DETECT PPE VIOLATIONS
# ----------------------------------
def detect_violations(results):
    violations = []  # Final list of violations to return
    detected_objects = []  # For logging all detected classes
    person_boxes = []  # Bounding boxes for 'Person'
    vest_boxes = []  # Bounding boxes for 'NO-Safety Vest'

    # Loop through all predictions
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Class index
            obj_name = result.names[cls]  # Class label (e.g., 'Person', 'NO-Mask')
            confidence = float(box.conf)  # Confidence score
            bbox = box.xyxy[0].tolist()  # Bounding box coordinates
            detected_objects.append(obj_name)

            # Separate detections by class
            if obj_name == 'Person':
                person_boxes.append(bbox)
            elif obj_name == 'NO-Safety Vest' and confidence >= CONFIDENCE_THRESHOLD:
                vest_boxes.append({'confidence': confidence, 'location': bbox})
            elif obj_name in ['NO-Hardhat', 'NO-Mask'] and confidence >= CONFIDENCE_THRESHOLD:
                violations.append({
                    'type': obj_name,
                    'confidence': confidence,
                    'location': bbox
                })

    # Assign each 'NO-Safety Vest' to only one person
    assigned = set()
    for vest in sorted(vest_boxes, key=lambda x: x['confidence'], reverse=True):
        best_iou = 0
        best_idx = -1
        for idx, person in enumerate(person_boxes):
            current_iou = iou(vest['location'], person)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx
        if best_iou > 0.3 and best_idx not in assigned:
            assigned.add(best_idx)
            violations.append({
                'type': 'NO-Safety Vest',
                'confidence': vest['confidence'],
                'location': vest['location']
            })

    # Log final results
    logging.info(f"Detected objects: {detected_objects}")
    logging.info(f"Filtered PPE violations: {violations}")

    return violations

# ----------------------------------
# 🔗 SAVE VIOLATIONS TO FIREBASE
# ----------------------------------
def save_violations_to_firebase(violations, timestamp, processed_image_url):
    violation_data = {
        'timestamp': timestamp,
        'violations': violations,
        'processed_image_url': processed_image_url
    }
    db.reference('violations').push(violation_data)  # Push data to Firebase
    logging.info("Violation data saved to Firebase.")

# ----------------------------------
# 📤 HANDLE IMAGE UPLOAD & PROCESSING
# ----------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.warning("No file selected")
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        logging.warning(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)  # Save file locally
        logging.info(f"File saved: {upload_path}")

        img = cv2.imread(upload_path)
        if img is None:
            logging.error(f"Failed to read image: {upload_path}")
            return jsonify({'error': 'Failed to read image'}), 400

        img = cv2.resize(img, (480, 480))  # Resize image for YOLO input
        logging.info(f"Processing image with YOLO model: {filename}")

        try:
            results = model(img, imgsz=480)  # Run YOLO inference
        except Exception as e:
            logging.error(f"Model inference failed: {str(e)}")
            return jsonify({'error': 'Model processing error'}), 500

        violations = detect_violations(results)  # Get PPE violations

        if violations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_filename = f"violation_{timestamp}_{filename}"
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

            annotated_img = results[0].plot()  # Annotate image
            cv2.imwrite(processed_path, annotated_img)  # Save annotated image
            logging.info(f"Processed image saved: {processed_path}")

            save_violations_to_firebase(
                violations,
                timestamp,
                f"http://{request.host}/processed/{processed_filename}"
            )

            return jsonify({
                'status': 'violation_detected',
                'violations': violations,
                'processed_image_url': f"http://{request.host}/processed/{processed_filename}",
                'timestamp': timestamp
            })

        else:
            return jsonify({
                'status': 'no_violation',
                'message': 'No PPE violations detected'
            })

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# ----------------------------------
# 📸 SERVE PROCESSED IMAGE FILE
# ----------------------------------
@app.route('/processed/<filename>')
def get_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)  # Return image file from processed directory

# ----------------------------------
# 🚀 RUN THE FLASK SERVER
# ----------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Start the app on all interfaces
