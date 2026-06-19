# COMP6721 Applied Artificial Intelligence — Course Project
**Student:** Himanshu Banwal  
**Student ID:** 40305160  
**Concordia University, Montreal — Summer 2026**

---

## GitHub Repository
https://github.com/happybanwal/COMP6721-Project

---

## Project Overview
Binary image classification of museum indoor/outdoor scenes using the MIT Places365 dataset.

**Phase 1:** Decision Tree, Gradient Boosting, Semi-Supervised Decision Tree (scikit-learn)  
**Phase 2:** Custom CNN and ResNet18 Transfer Learning (PyTorch)

---

## Dataset
MIT Places365 — Museum Indoor/Outdoor  
Download: http://places.csail.mit.edu/browser.html  
Place images in:
dataset/Training/museum-indoor/

dataset/Training/museum-outdoor/

dataset/Museum_Test/museum-indoor/

dataset/Museum_Test/museum-outdoor/

---

## Requirements
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow numpy
```

---

## Phase 2 — How to Train
```bash
cd phase2
python3 train.py
```
Trained models are saved to `phase2/saved_models/`

---

## Phase 2 — How to Evaluate
```bash
cd phase2
python3 evaluate.py
```

---

## Phase 2 — Run on a Single Image (Demo Mode)
```bash
cd phase2
python3 predict.py path/to/image.jpg resnet
python3 predict.py path/to/image.jpg cnn
```

---

## Phase 1 — How to Run
```bash
cd phase1/code
python3 pipeline.py
```

---

## Results Summary

| Model | Accuracy | F1 |
|---|---|---|
| Decision Tree | 0.665 | 0.656 |
| Gradient Boosting | 0.730 | 0.716 |
| Semi-Supervised DT | 0.625 | 0.623 |
| Custom CNN | 0.928 | 0.926 |
| ResNet18 | **0.969** | **0.968** |


