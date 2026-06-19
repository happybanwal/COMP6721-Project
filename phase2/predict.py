import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

from model_cnn import CustomCNN
from model_resnet import build_resnet

# Class names (alphabetical - same as ImageFolder)
CLASS_NAMES = ['museum-indoor', 'museum-outdoor']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transform for single image ─────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Load model ─────────────────────────────────────────────────
def load_model(model_type="resnet"):
    if model_type == "cnn":
        model = CustomCNN(num_classes=2)
        path  = "saved_models/custom_cnn_best.pth"
    else:
        model = build_resnet(num_classes=2)
        path  = "saved_models/resnet18_best.pth"

    model.load_state_dict(torch.load(path,
                          map_location=device,
                          weights_only=True))
    model.to(device)
    model.eval()
    return model

# ── Predict single image ───────────────────────────────────────
def predict(image_path, model_type="resnet"):
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    # unsqueeze(0) adds batch dimension: [3,224,224] → [1,3,224,224]

    model = load_model(model_type)

    with torch.no_grad():
        outputs = model(tensor)              # shape: [1, 2]
        probs   = torch.softmax(outputs, dim=1)  # convert to probabilities
        conf, predicted = probs.max(1)       # highest probability class

    label = CLASS_NAMES[predicted.item()]
    confidence = conf.item() * 100

    print(f"\nImage:      {os.path.basename(image_path)}")
    print(f"Model:      {model_type}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nAll probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[0][i].item()*100:.2f}%")

    return label, confidence


# ── Run from command line ──────────────────────────────────────
# Usage: python3 predict.py <image_path> <model_type>
# Example: python3 predict.py test.jpg resnet
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <image_path> [cnn|resnet]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "resnet"

    predict(image_path, model_type)