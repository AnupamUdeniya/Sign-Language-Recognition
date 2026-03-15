import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as transforms
from scripts.model import get_model

# Device
device = torch.device("cpu")

# Load model
model = get_model(29)
model.load_state_dict(torch.load("best_asl_hybrid.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Classes
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space", "nothing"]


def predict(img):

    # Convert numpy image to PIL
    img = Image.fromarray(img).convert("RGB")

    # Transform
    img = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    result = f"{classes[pred]}  (confidence: {confidence:.2f})"

    return result


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload ASL Hand Sign"),
    outputs=gr.Textbox(label="Prediction"),
    title="ASL Sign Language Recognition",
    description="Upload a hand sign image and the model will predict the ASL alphabet."
)

demo.launch()