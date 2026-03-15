import base64
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

    model = get_model(
        num_classes=len(CLASS_NAMES),
        pretrained_backbone=False
    ).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    return _model


def decode_data_url(image_data):
    if not isinstance(image_data, str) or "," not in image_data:
        raise ValueError("Invalid image data URL.")

    _, encoded = image_data.split(",", 1)
    return base64.b64decode(encoded)


def read_request_image():
    uploaded_file = request.files.get("image")
    if uploaded_file is not None:
        return uploaded_file.read()

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image") or payload.get("data_url")

    if image_data:
        return decode_data_url(image_data)

    raise ValueError("No image file or data URL received.")


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
@app.get("/api/health")
def health():
    try:
        load_model()
        return jsonify({
            "status": "ok",
            "backend": "local-pytorch",
            "device": str(DEVICE)
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": str(exc)
        }), 500


@app.post("/predict")
@app.post("/api/predict")
def predict():
    try:
        image_bytes = read_request_image()
        result = predict_image(image_bytes)
        result["backend"] = "local-pytorch"
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
