from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import requests

app = FastAPI()

# =========================
# LOAD YOLO
# =========================
model = YOLO("yolov8n.pt")

# =========================
# LAZY LOAD FLAN-T5
# =========================
generator = None

def get_generator():
    global generator
    if generator is None:
        from transformers import pipeline
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-small"
        )
    return generator

# =========================
# YOLO DETECTION
# =========================
def detect_objects(image_path):
    results = model(image_path)
    objects = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            objects.append(model.names[cls_id])

    return objects

# =========================
# LLaVA (COLAB API)
# =========================
def get_caption(image_path):
    url = "https://helps-campbell-local-cube.trycloudflare.com/predict"

    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files, timeout=160)

        if response.status_code != 200:
            return "Caption service error"

        return response.json().get("caption", "No caption")

    except Exception:
        return "Caption service unavailable"

# =========================
# AI ANALYSIS (FLAN-T5)
# =========================
def generate_ai_analysis(objects, caption):

    try:
        gen = get_generator()

        prompt = f"""
        You are a traffic accident analysis system.

        Scene: {caption}
        Objects: {', '.join(objects)}

        Provide:
        Cause:
        Precautions:
        """

        result = gen(prompt, max_length=80)[0]['generated_text']

        return result

    except Exception:
        return "AI analysis temporarily unavailable."

# =========================
# API ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = "temp.jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 1: Object detection
    objects = detect_objects(file_path)

    # Step 2: Caption (LLaVA)
    caption = get_caption(file_path)

    # Step 3: AI reasoning
    analysis = generate_ai_analysis(objects, caption)

    # Final response
    return {
        "objects": objects,
        "caption": caption,
        "analysis": analysis
    }