import torch
import torch.nn as nn
from torch.optim import Adam
from data_pipeline import train_loader, val_loader, CLASS_NAMES
from model_cnn import CustomCNN
from model_resnet import build_resnet
import os
import json

# ── Device ─────────────────────────────────────────────────────
# Use GPU if available, otherwise CPU
# On Colab this will be 'cuda', locally 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Training function ──────────────────────────────────────────
def train_model(model, train_loader, val_loader,
                num_epochs=10, learning_rate=0.001,
                model_name="model"):
    """
    Train a model and validate after each epoch.

    Args:
        model:         the CNN or ResNet model
        train_loader:  batches of training images
        val_loader:    batches of validation images
        num_epochs:    how many full passes through training data
        learning_rate: step size for weight updates
        model_name:    used for saving the model file
    """

    model = model.to(device)  # move model to GPU/CPU

    # CrossEntropyLoss: measures how wrong predictions are
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: updates weights using gradients
    # only pass parameters that require gradients
    # (frozen ResNet layers are automatically excluded)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # Store history for plotting later
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    best_val_acc = 0.0   # track best model

    for epoch in range(num_epochs):

        # ── Training phase ─────────────────────────────────────
        model.train()   # sets model to training mode
                        # (enables dropout, batch norm updates)
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)   # move batch to GPU/CPU
            labels = labels.to(device)

            # Step 1: Forward pass
            outputs = model(images)      # shape: [32, 2]

            # Step 2: Calculate loss
            loss = criterion(outputs, labels)

            # Step 3: Zero gradients from previous batch
            # (PyTorch accumulates gradients by default)
            optimizer.zero_grad()

            # Step 4: Backward pass — compute gradients
            loss.backward()

            # Step 5: Update weights using gradients
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)   # take class with higher score
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1} | "
                      f"Loss: {loss.item():.4f}")

        train_acc  = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # ── Validation phase ───────────────────────────────────
        model.eval()    # sets model to evaluation mode
                        # (disables dropout, freezes batch norm)
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():   # don't compute gradients during validation
                                # saves memory and speeds things up
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # ── Save best model ────────────────────────────────────
        # We save the model with the best validation accuracy
        # not the last epoch — last epoch may have overfit
        os.makedirs("saved_models", exist_ok=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f"saved_models/{model_name}_best.pth")
            print(f"  ✓ Best model saved (val_acc: {val_acc:.2f}%)")

        # Store history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        print("-" * 60)

    # Save training history for plotting
    with open(f"saved_models/{model_name}_history.json", "w") as f:
        json.dump(history, f)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    return history


# ── Run training ───────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("Training Custom CNN")
    print("=" * 60)
    cnn_model = CustomCNN(num_classes=2, dropout_rate=0.5)
    cnn_history = train_model(
        cnn_model, train_loader, val_loader,
        num_epochs=10,
        learning_rate=0.001,
        model_name="custom_cnn"
    )

    print("\n" + "=" * 60)
    print("Training ResNet18")
    print("=" * 60)
    resnet_model = build_resnet(num_classes=2, freeze_backbone=True)
    resnet_history = train_model(
        resnet_model, train_loader, val_loader,
        num_epochs=10,
        learning_rate=0.001,
        model_name="resnet18"
    )