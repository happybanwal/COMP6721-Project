"""
COMP6721 Applied AI - Phase 1 Demo Prediction Script
Usage: python predict.py path/to/image.jpg
"""

import os
import sys
import numpy as np
from PIL import Image
import pickle

IMG_SIZE = (32, 32)
MODEL_PATH = "best_model.pkl"

# ─────────────────────────────────────────────
#  LOAD AND PREPROCESS A SINGLE IMAGE
# ─────────────────────────────────────────────
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    features = np.array(img, dtype=np.float32).flatten() / 255.0
    return features.reshape(1, -1)

# ─────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────
def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
        print("Please run train_and_save.py first to train and save the model.")
        sys.exit(1)

    # Load saved model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Preprocess image
    features = preprocess_image(image_path)

    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    label = "museum-indoor" if prediction == 0 else "museum-outdoor"
    confidence = probability[prediction] * 100

    print("\n" + "=" * 45)
    print("  PREDICTION RESULT")
    print("=" * 45)
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {label.upper()}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"  P(indoor)  : {probability[0]*100:.2f}%")
    print(f"  P(outdoor) : {probability[1]*100:.2f}%")
    print("=" * 45)

    return label, confidence


# ─────────────────────────────────────────────
#  TRAIN AND SAVE BEST MODEL
# ─────────────────────────────────────────────
def train_and_save():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import f1_score

    TRAIN_DIR = os.path.expanduser("~/Desktop/ai/dataset/Training")

    print("Loading training data...")
    X, y = [], []
    class_map = {"museum-indoor": 0, "museum-outdoor": 1}

    for class_name, label in class_map.items():
        class_path = os.path.join(TRAIN_DIR, class_name)
        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Loading {len(files)} images from '{class_name}' ...")
        for fname in files:
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).convert('L')
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                X.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                y.append(label)
            except:
                pass

    X, y = np.array(X), np.array(y)
    print(f"  Total: {len(X)} samples")

    # Train best model (GB n=50 had best F1)
    print("\nTraining best model (Gradient Boosting n=50)...")
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                        learning_rate=0.1, random_state=42)
    model.fit(X, y)

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"  Model saved to '{MODEL_PATH}'")
    print("  Ready for demo predictions!")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "--train":
            train_and_save()
        else:
            # Predict on provided image
            image_path = sys.argv[1]
            if not os.path.exists(image_path):
                print(f"[ERROR] Image not found: {image_path}")
                sys.exit(1)
            predict(image_path)
    else:
        print("Usage:")
        print("  Train and save model : python predict.py --train")
        print("  Predict on an image  : python predict.py path/to/image.jpg")
