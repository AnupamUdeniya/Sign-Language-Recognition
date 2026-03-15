import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from scripts.model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = sorted([
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
])

MODEL = get_model(
    num_classes=len(CLASS_NAMES),
    pretrained_backbone=False
)
MODEL.load_state_dict(torch.load("best_asl_hybrid.pth", map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def predict(image_array):
    if image_array is None:
        return "nothing (confidence: 0.00)"

    image = Image.fromarray(image_array).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item()

    return f"{predicted_label} (confidence: {confidence:.2f})"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload ASL Hand Sign"),
    outputs=gr.Textbox(label="Prediction"),
    title="ASL Sign Language Recognition",
    description="Upload a hand sign image and the model will predict the ASL alphabet."
)

demo.launch()
