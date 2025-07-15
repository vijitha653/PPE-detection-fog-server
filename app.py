from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
from datetime import datetime
from ultralytics import YOLO
import logging
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import numpy as np

# -----------------------------
# 🔍 CONFIGURE LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# 🚀 INITIALIZE FLASK APP
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# 🛠️ CONFIGURATION
# -----------------------------
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_PATH = 'model/best.onnx'
FIREBASE_CREDENTIALS = 'serviceAccountKey.json'
CONFIDENCE_THRESHOLD = 0.4

# -----------------------------
# ☁️ INITIALIZE FIREBASE
# -----------------------------
if not os.path.exists(FIREBASE_CREDENTIALS):
    logging.error("Firebase credentials file missing!")
    raise FileNotFoundError("Firebase credentials file not found.")

cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ppe-detection-cloud-default-rtdb.firebaseio.com/'
})

# -----------------------------
# 📁 ENSURE FOLDER STRUCTURE
# -----------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# -----------------------------
# 🤖 LOAD YOLO MODEL
# -----------------------------
try:
    model = YOLO(MODEL_PATH)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

# -----------------------------
# 🔎 FILE VALIDATION
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# 🔢 IoU CALCULATION
# -----------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

# -----------------------------
# 🚨 DETECT PPE VIOLATIONS
# -----------------------------
def detect_violations(results):
    violations = []
    detected_objects = []

    person_boxes = []
    vest_boxes = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            obj_name = result.names[cls]
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()
            detected_objects.append(obj_name)

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

    # Deduplicate NO-Safety Vest violations per person
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

    logging.info(f"Detected objects: {detected_objects}")
    logging.info(f"Filtered PPE violations: {violations}")

    return violations

# -----------------------------
# 🔗 SAVE TO FIREBASE
# -----------------------------
def save_violations_to_firebase(violations, timestamp, processed_image_url):
    violation_data = {
        'timestamp': timestamp,
        'violations': violations,
        'processed_image_url': processed_image_url
    }
    db.reference('violations').push(violation_data)
    logging.info("Violation data saved to Firebase.")

# -----------------------------
# 📤 FILE UPLOAD ROUTE
# -----------------------------
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
        file.save(upload_path)
        logging.info(f"File saved: {upload_path}")

        img = cv2.imread(upload_path)
        if img is None:
            logging.error(f"Failed to read image: {upload_path}")
            return jsonify({'error': 'Failed to read image'}), 400

        img = cv2.resize(img, (480, 480))
        logging.info(f"Processing image with YOLO model: {filename}")

        try:
            results = model(img, imgsz=480)
        except Exception as e:
            logging.error(f"Model inference failed: {str(e)}")
            return jsonify({'error': 'Model processing error'}), 500

        violations = detect_violations(results)

        if violations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_filename = f"violation_{timestamp}_{filename}"
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

            annotated_img = results[0].plot()
            cv2.imwrite(processed_path, annotated_img)
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

# -----------------------------
# 📸 SERVE PROCESSED IMAGE
# -----------------------------
@app.route('/processed/<filename>')
def get_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

# -----------------------------
# 🚀 START FLASK APP
# -----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
