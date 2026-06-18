import torch
import torch.nn as nn
from torchvision import models

def build_resnet(num_classes=2, freeze_backbone=True):
    """
    Load pretrained ResNet18 and adapt it for binary classification.

    Args:
        num_classes:     number of output classes (2: indoor/outdoor)
        freeze_backbone: if True, freeze all layers except the final FC
                         if False, fine-tune the entire network
    """

    # ── Load pretrained ResNet18 ───────────────────────────────
    # weights='IMAGENET1K_V1' downloads weights trained on ImageNet
    # The model already knows how to detect edges, textures, shapes
    model = models.resnet18(weights='IMAGENET1K_V1')

    # ── Freeze all layers ──────────────────────────────────────
    # requires_grad=False means these weights won't be updated
    # during backpropagation — we preserve ImageNet knowledge
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ── Replace the final layer ────────────────────────────────
    # ResNet18's original final layer: Linear(512 → 1000 classes)
    # We replace it with: Linear(512 → 2 classes)
    # This new layer has requires_grad=True by default
    # so only this layer gets trained
    in_features = model.fc.in_features   # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),     # compress 512 → 256
        nn.ReLU(),
        nn.Dropout(p=0.5),               # regularization
        nn.Linear(256, num_classes)      # final: 256 → 2
    )

    return model


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_resnet(num_classes=2, freeze_backbone=True)

    # Count trainable vs frozen parameters
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")

    # Simulate one batch
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")   # → [4, 2]
    print("ResNet OK")