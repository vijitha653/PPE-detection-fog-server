from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import uvicorn

# ========== Firebase Configuration ==========
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase (place near top after other imports)
cred = credentials.Certificate("serviceAccountKey.json")  # File in root folder
firebase_admin.initialize_app(cred, {
    'storageBucket': "your-project-id.appspot.com"  # ← Replace with your bucket
})

def upload_to_firebase(local_path):
    """Uploads files to Firebase Storage and returns public URL"""
    bucket = storage.bucket()
    blob = bucket.blob(f"violations/{os.path.basename(local_path)}")
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url
# ========== End Firebase Config ==========

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = YOLO("model/best.onnx")

@app.post("/upload")
async def detect_ppe(file: UploadFile = File(...)):
    try:
        # 1. Process image
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "Invalid image file")

        # 2. Run detection
        results = model(image)
        violations = [
            model.names[int(box.cls)] 
            for box in results[0].boxes
            if model.names[int(box.cls)] in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
        ]

        # 3. Handle violations
        if violations:
            # Save locally
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_path = f"cloud_storage/violation_{timestamp}.jpg"  # Using your existing folder
            cv2.imwrite(local_path, image)
            
            # Upload to Firebase
            cloud_url = upload_to_firebase(local_path)
            
            return {
                "status": "VIOLATION_DETECTED",
                "violations": violations,
                "local_path": local_path,
                "cloud_url": cloud_url
            }
            
        return {"status": "NO_VIOLATIONS"}
        
    except Exception as e:
        raise HTTPException(500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)