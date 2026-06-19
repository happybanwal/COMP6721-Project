from data_pipeline import train_loader, val_loader
from model_cnn import CustomCNN
from model_resnet import build_resnet
from train import train_model

# Experiment 1: Lower learning rate (0.0001 vs default 0.001)
print("ABLATION 1: CNN - Learning Rate 0.0001")
model = CustomCNN(num_classes=2)
train_model(model, train_loader, val_loader,
            num_epochs=5, learning_rate=0.0001,
            model_name="cnn_lr0001")

# Experiment 2: CNN with only 2 blocks (shallower)
print("ABLATION 2: ResNet - Unfrozen backbone (full fine-tuning)")
model = build_resnet(num_classes=2, freeze_backbone=False)
train_model(model, train_loader, val_loader,
            num_epochs=5, learning_rate=0.0001,
            model_name="resnet_unfrozen")