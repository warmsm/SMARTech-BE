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
        # FIXED: Pointing to best.pt in the root directory
        model_path = os.path.join(base_path, "best.pt")
        print(f"Loading YOLO model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        MODELS["logo"] = YOLO(model_path)
    
    if MODELS["ocr"] is None:
        print("Loading docTR model...")
        MODELS["ocr"] = get_doctr_model()
    
    return MODELS["logo"], MODELS["ocr"]

@app.get("/")
def health_check():
    # Keep this for monitoring Render's lazy memory mode
    return {"status": "online", "memory_mode": "lazy"}

@app.post("/audit")
async def audit_pubmat(
    file: UploadFile = File(...), 
    post_type: str = Form(...), 
    collaborators: str = Form("[]") 
):
    try:
        # 1. Load models (only if they aren't loaded yet)
        logo_model, _ = load_models()

        # 2. Process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format or corrupted file."}

        try:
            collab_list = json.loads(collaborators)
        except:
            collab_list = []

        # 3. Run audit
        # Wrapped in a try-except to catch 500 errors in Swagger
        report, _ = generate_report(img, logo_model, post_type, collab_list)

        return report

    except Exception as e:
        # This will now show the SPECIFIC error in Swagger UI instead of just '500'
        return {"status": "error", "message": f"Audit failed: {str(e)}"}