import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

from data_pipeline import test_loader, demo_loader, CLASS_NAMES
from model_cnn import CustomCNN
from model_resnet import build_resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load a saved model ─────────────────────────────────────────
def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ── Get predictions ────────────────────────────────────────────
def get_predictions(model, loader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

# ── Print metrics ──────────────────────────────────────────────
def evaluate_model(model, loader, model_name):
    labels, preds = get_predictions(model, loader)

    acc  = accuracy_score(labels, preds) * 100
    prec = precision_score(labels, preds) * 100
    rec  = recall_score(labels, preds) * 100
    f1   = f1_score(labels, preds) * 100

    print(f"\n{'='*50}")
    print(f"Results: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall:    {rec:.2f}%")
    print(f"F1 Score:  {f1:.2f}%")
    print(f"\n{classification_report(labels, preds, target_names=CLASS_NAMES)}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/confusion_matrix_{model_name}.png")
    plt.close()
    print(f"Confusion matrix saved to results/")

    return {"accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}

# ── Plot training curves ───────────────────────────────────────
def plot_history(model_name):
    path = f"saved_models/{model_name}_history.json"
    with open(path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"],   label="Val Acc")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"],   label="Val Loss")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/{model_name}_curves.png")
    plt.close()
    print(f"Training curves saved to results/")


# ── Run evaluation ─────────────────────────────────────────────
if __name__ == "__main__":

    # Custom CNN
    cnn = load_model(CustomCNN(num_classes=2),
                     "saved_models/custom_cnn_best.pth")
    cnn_metrics = evaluate_model(cnn, test_loader, "custom_cnn")
    plot_history("custom_cnn")

    # ResNet
    resnet = load_model(build_resnet(num_classes=2),
                        "saved_models/resnet18_best.pth")
    resnet_metrics = evaluate_model(resnet, test_loader, "resnet18")
    plot_history("resnet18")

    # Save comparison
    comparison = {
        "custom_cnn": cnn_metrics,
        "resnet18":   resnet_metrics
    }
    with open("results/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("\nComparison saved to results/comparison.json")