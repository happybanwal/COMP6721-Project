import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(CustomCNN, self).__init__()

        # ── Block 1 ────────────────────────────────────────────
        # Input: 3 channels (RGB), output: 32 feature maps
        # Kernel 3×3 slides across image detecting low-level features
        # padding=1 keeps spatial size same after conv
        # After MaxPool: 224×224 → 112×112
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── Block 2 ────────────────────────────────────────────
        # Input: 32 feature maps, output: 64 feature maps
        # Detects more complex patterns by combining block1 features
        # After MaxPool: 112×112 → 56×56
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── Block 3 ────────────────────────────────────────────
        # Input: 64 feature maps, output: 128 feature maps
        # Detects high-level features (structures, textures)
        # After MaxPool: 56×56 → 28×28
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── Classifier ─────────────────────────────────────────
        # Flatten: 28×28×128 = 100352 values → 1D vector
        # FC1: compress 100352 → 256
        # Dropout: randomly zeros 50% of neurons during training
        #          forces network not to rely on any single neuron
        #          reduces overfitting
        # FC2: 256 → 2 (one score per class: indoor, outdoor)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape coming in: [batch_size, 3, 224, 224]
        x = self.block1(x)   # → [batch_size, 32, 112, 112]
        x = self.block2(x)   # → [batch_size, 64, 56, 56]
        x = self.block3(x)   # → [batch_size, 128, 28, 28]
        x = self.classifier(x)  # → [batch_size, 2]
        return x


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    model = CustomCNN(num_classes=2)
    print(model)

    # Simulate one batch: 4 images, 3 channels, 224×224
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")   # should be [4, 2]
    print("CustomCNN OK")