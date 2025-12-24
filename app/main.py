from fastapi import FastAPI, UploadFile, File
from app.model import model, DEVICE
from app.utils import preprocess_image, predict, CLASSES

app = FastAPI(
    title="Pap Smear Cell Classifier",
    description="EfficientNet-B0 based cytology screening model",
    version="1.0"
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def classify_cell(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    img_tensor = preprocess_image(file.file)
    result = predict(model, img_tensor, DEVICE)
    return {
        "prediction": result['prediction'],
        "confidence": result["confidence"],
        "classes": CLASSES
    }
@app.get("/info")
def info():
    return {
        "model": "EfficientNet-B0",
        "dataset": "SIPaKMeD",
        "num_classes": 5,
        "input": "224x224 RGB cropped cervical cell image",
        "framework": "PyTorch",
        "device": str(DEVICE),
        "test_accuracy": 0.9326,
        "disclaimer": (
            "This system is intended for research and screening assistance only. "
            "It must not be used for medical diagnosis."
        )
    }
