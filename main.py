import os
from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
from ultralytics import YOLO
import json
from checker import generate_report, get_doctr_model

app = FastAPI()

MODELS = {
    "logo": None,
    "ocr": None
}

def load_models():
    """Helper to load models with absolute path resolution."""
    # This finds the exact absolute path of the folder where main.py sits
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    if MODELS["logo"] is None:
        # Check both the root and the weights folder just in case
        potential_paths = [
            os.path.join(base_path, "best.pt"),
            os.path.join(base_path, "weights", "best.pt"),
            "/opt/render/project/src/best.pt" # Render's standard absolute path
        ]
        
        model_path = None
        for path in potential_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            print(f"✅ Success: Loading YOLO model from {model_path}")
            MODELS["logo"] = YOLO(model_path)
        else:
            # This specific error will now show up in your Swagger UI
            raise FileNotFoundError(f"❌ Model weights 'best.pt' not found. Searched: {potential_paths}")
    
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
    try:
        logo_model, _ = load_models()

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        try:
            collab_list = json.loads(collaborators)
        except:
            collab_list = []

        # Run audit logic
        report, _ = generate_report(img, logo_model, post_type, collab_list)
        return report

    except Exception as e:
        # This catch-all ensures your presentation doesn't just show '500'
        return {
            "status": "error", 
            "message": f"Critical Audit Failure: {str(e)}",
            "tip": "Check Render logs for memory overflow if the error persists."
        }