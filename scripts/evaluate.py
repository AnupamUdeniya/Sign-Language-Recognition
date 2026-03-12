import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report

try:
    from scripts.asl_dataset import get_dataloaders
    from scripts.model import get_model
except ModuleNotFoundError:
    from asl_dataset import get_dataloaders
    from model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    print("Using device:", DEVICE)

    _, _, test_loader = get_dataloaders(
        "ASL_Alphabet_Dataset",
        batch_size=64
    )

    model = get_model(29).to(DEVICE)
    model.load_state_dict(torch.load("best_asl_hybrid.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()

    # ================= Accuracy =================
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100 * np.mean(all_preds == all_labels)
    print("\nTest Accuracy: {:.2f}%".format(accuracy))

    # ================= Classification Report =================
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

    # ================= Confusion Matrix =================
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ================= Efficiency Metrics =================
    total_time = end_time - start_time
    num_images = len(test_loader.dataset)

    avg_time = total_time / num_images
    fps = 1 / avg_time

    print("\nTotal Inference Time: {:.4f} seconds".format(total_time))
    print("Average Time per Image: {:.4f} ms".format(avg_time * 1000))
    print("FPS: {:.2f}".format(fps))


if __name__ == "__main__":
    main()
