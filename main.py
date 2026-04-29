import os
from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
from ultralytics import YOLO
import json
from checker import generate_report, get_doctr_model

app = FastAPI()

# Hold models in memory after first use
MODELS = {
    "logo": None,
    "ocr": None
}

def load_models():
    """Helper to load models only when needed."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    if MODELS["logo"] is None:
        print("Loading YOLO model...")
        model_path = os.path.join(base_path, "weights", "best.pt")
        MODELS["logo"] = YOLO(model_path)
    
    if MODELS["ocr"] is None:
        print("Loading docTR model...")
        MODELS["ocr"] = get_doctr_model()
    
    return MODELS["logo"], MODELS["ocr"]

@app.get("/")
def health_check():
    return {"status": "online", "memory_mode": "lazy"}

@app.post("/audit")
async def audit_pubmat(
    file: UploadFile = File(...), 
    post_type: str = Form(...), 
    collaborators: str = Form("[]") 
):
    # 1. Load models (only if they aren't loaded yet)
    logo_model, _ = load_models()

    # 2. Process image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image format"}

    try:
        collab_list = json.loads(collaborators)
    except:
        collab_list = []

    # 3. Run audit
    # Note: checker.py's generate_report internally calls get_doctr_model()
    # so we just pass the logo_model here.
    report, _ = generate_report(img, logo_model, post_type, collab_list)

    return report