import torch
import cv2
from PIL import Image
from torchvision import transforms

try:
    from scripts.model import get_model
except ModuleNotFoundError:
    from model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (must match ImageFolder order)
class_names = sorted([
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def load_model():
    model = get_model(29).to(DEVICE)
    model.load_state_dict(torch.load("best_asl_hybrid.pth", map_location=DEVICE))
    model.eval()
    return model


def main():
    model = load_model()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            img = transform(pil_image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(img)
                _, pred = torch.max(output, 1)

            label = class_names[pred.item()]

            cv2.putText(frame, label, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

            cv2.imshow("ASL Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
