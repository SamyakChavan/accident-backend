from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # ADD THIS IMPORT
from ultralytics import YOLO
import shutil
import requests

app = FastAPI()

# ============================================
# ADD CORS MIDDLEWARE - COPY THIS ENTIRE BLOCK
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ============================================

# Load YOLO model
model = YOLO("yolov8n.pt")

# YOLO detection
def detect_objects(image_path):
    results = model(image_path)
    objects = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            objects.append(model.names[cls_id])

    return objects

# LLaVA (Colab)
def get_caption(image_path):
    url = "https://destinations-creator-misc-columbus.trycloudflare.com/predict"

    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files, timeout=160)

        if response.status_code != 200:
            return f"Error from model API: {response.text}"

        return response.json().get("caption", "No caption")

    except Exception as e:
        return f"Connection error: {str(e)}"

# Cause
def generate_cause(objects):
    if objects.count("car") >= 2:
        return "Vehicle collision likely due to overspeeding."
    elif "person" in objects and "car" in objects:
        return "Pedestrian accident possible."
    else:
        return "Cause unclear."

# Precautions
def generate_precautions():
    return [
        "Maintain safe distance",
        "Follow traffic rules",
        "Avoid overspeeding"
    ]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = "temp.jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    objects = detect_objects(file_path)
    caption = get_caption(file_path)
    cause = generate_cause(objects)
    precautions = generate_precautions()

    # Clean up temp file
    import os
    if os.path.exists(file_path):
        os.remove(file_path)

    return {
        "objects": objects,
        "caption": caption,
        "cause": cause,
        "precautions": precautions
    }

# Optional: Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "cors": "enabled", "message": "API is ready"}

# Optional: Add root endpoint
@app.get("/")
async def root():
    return {"message": "AccidentSense API is running", "cors_enabled": True}