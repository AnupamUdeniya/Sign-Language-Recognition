import io
import os
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.model import get_model

CLASS_NAMES = sorted([
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "best_asl_hybrid.pth"
WEB_DIR = ROOT / "web"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
_model = None


def load_model():
    global _model

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = get_model(num_classes=len(CLASS_NAMES), pretrained_backbone=False).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    return _model


def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    model = load_model()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_values, top_indices = torch.topk(probabilities, k=min(5, len(CLASS_NAMES)))
    top_predictions = []

    for score, index in zip(top_values.tolist(), top_indices.tolist()):
        top_predictions.append({
            "label": CLASS_NAMES[index],
            "confidence": round(score * 100, 2)
        })

    best_index = top_indices[0].item()
    return {
        "label": CLASS_NAMES[best_index],
        "confidence": round(top_values[0].item() * 100, 2),
        "top_predictions": top_predictions
    }


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health():
    try:
        load_model()
        return jsonify({"status": "ok", "device": str(DEVICE)})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image file received."}), 400

    try:
        result = predict_image(file.read())
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
