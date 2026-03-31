from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import requests

app = FastAPI()

# Load YOLO 
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
    url = "https://stable-suite-insertion-track.trycloudflare.com/predict"  # 🔴 your ngrok

    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)

    return response.json().get("caption", "No caption")

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

    return {
        "objects": objects,
        "caption": caption,
        "cause": cause,
        "precautions": precautions
    }
    # test change