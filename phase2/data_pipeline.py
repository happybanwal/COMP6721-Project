import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os

# ── Paths ──────────────────────────────────────────────────────
TRAIN_DIR = os.path.expanduser("~/Desktop/ai/dataset/Training")
TEST_DIR  = os.path.expanduser("~/Desktop/ai/dataset/Museum_Test")

# ── Transforms ─────────────────────────────────────────────────

# Training: augmentation helps the model generalize
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),         # CNN needs fixed input size
    transforms.RandomHorizontalFlip(),     # mirror image — still same class
    transforms.RandomRotation(10),         # slight rotation — still same class
    transforms.ToTensor(),                 # PIL image → tensor, 0-255 → 0.0-1.0
    transforms.Normalize(                  # center values using ImageNet stats
        mean=[0.485, 0.456, 0.406],        # (matches ResNet preprocessing too)
        std=[0.229, 0.224, 0.225]
    )
])

# Val/Test: no augmentation — we want deterministic, consistent evaluation
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Load full Training dataset ──────────────────────────────────

# ImageFolder reads subfolder names as class labels (alphabetical order)
# museum-indoor → 0, museum-outdoor → 1
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)

CLASS_NAMES = full_dataset.classes   # ['museum-indoor', 'museum-outdoor']
print(f"Classes: {CLASS_NAMES}")
print(f"Total training images: {len(full_dataset)}")

# ── Split 70 / 15 / 15 ─────────────────────────────────────────

total      = len(full_dataset)           # 10,000
train_size = int(0.70 * total)           # 7,000
val_size   = int(0.15 * total)           # 1,500
test_size  = total - train_size - val_size  # 1,500 (remainder avoids rounding)

# manual_seed(42) makes the split identical every run → reproducible results
generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator
)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

# ── Apply eval transforms to val and test ──────────────────────
# Problem: random_split shares the parent dataset's transform
# We need train → augmentation, val/test → no augmentation
# Solution: a small wrapper that overrides the transform

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get the original file path and label from the parent dataset
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert("RGB")   # load image from disk
        return self.transform(img), label        # apply our transform

val_set  = TransformSubset(val_set,  eval_transforms)
test_set = TransformSubset(test_set, eval_transforms)
# train_set keeps train_transforms from full_dataset automatically

# ── Professor's fixed test set (for demo) ──────────────────────
demo_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transforms)
print(f"Demo test images: {len(demo_dataset)}")  # → 200

# ── DataLoaders ────────────────────────────────────────────────
BATCH_SIZE = 32

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,       # randomize order each epoch
    num_workers=2       # parallel CPU loading while GPU trains
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,      # no need to shuffle evaluation
    num_workers=2
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

demo_loader = DataLoader(
    demo_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# ── Sanity check ───────────────────────────────────────────────
if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")   # → [32, 3, 224, 224]
    print(f"Label sample: {labels[:8]}")     # → mix of 0s and 1s
    print("Data pipeline OK")
    
