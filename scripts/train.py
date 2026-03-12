import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    from scripts.asl_dataset import get_dataloaders
    from scripts.model import get_model
except ModuleNotFoundError:
    from asl_dataset import get_dataloaders
    from model import get_model


def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    writer = SummaryWriter("runs/asl_hybrid")

    train_loader, val_loader, _ = get_dataloaders(
        "ASL_Alphabet_Dataset",
        batch_size=32
    )

    model = get_model(29).to(DEVICE)

    for param in model.cnn.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW([
        {"params": model.cnn.parameters(), "lr": 1e-4},
        {"params": model.transformer.parameters(), "lr": 3e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4},
        {"params": model.fusion.parameters(), "lr": 3e-4},
        {"params": model.patch_proj.parameters(), "lr": 3e-4},
        {"params": model.cnn_proj.parameters(), "lr": 3e-4},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=40
    )

    EPOCHS = 40
    best_val_acc = 0
    patience = 7
    early_stop_counter = 0

    for epoch in range(EPOCHS):

        # -------- Gradual Unfreezing --------
        if epoch == 3:
            print("Unfreezing layer4...")
            for name, param in model.cnn.named_parameters():
                if "layer4" in name:
                    param.requires_grad = True

        if epoch == 8:
            print("Unfreezing full backbone...")
            for param in model.cnn.parameters():
                param.requires_grad = True

        # ================= TRAIN =================
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            # Gradient clipping (important for transformer stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        scheduler.step()

        # -------- TensorBoard Logging --------
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # -------- Save Best Model --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_asl_hybrid.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # -------- Early Stopping --------
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print("\nBest Validation Accuracy:", best_val_acc)
    writer.close()


if __name__ == "__main__":
    main()
