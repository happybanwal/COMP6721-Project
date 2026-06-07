"""
COMP6721 Applied AI - Phase 1 Pipeline
Supervised Decision Tree + Boosting + Semi-Supervised Decision Tree
Binary Classification: museum-indoor vs museum-outdoor
"""

import os
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG  — adjust paths if needed
# ─────────────────────────────────────────────
TRAIN_DIR      = os.path.expanduser("~/Desktop/ai/dataset/Training")
VAL_DIR        = os.path.expanduser("~/Desktop/ai/dataset/Museum_Validation")
IMG_SIZE       = (32, 32)          # resize target
RESULTS_DIR    = "phase1_results"  # folder where plots/results are saved
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  1. DATA LOADING
# ─────────────────────────────────────────────
def load_images(root_dir, img_size=IMG_SIZE):
    """Load images from root_dir/museum-indoor and root_dir/museum-outdoor.
    Returns X (n_samples, H*W) as float32 and y (n_samples,) as int."""
    X, y = [], []
    class_map = {"museum-indoor": 0, "museum-outdoor": 1}

    for class_name, label in class_map.items():
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"  [WARNING] Folder not found: {class_path}")
            continue
        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Loading {len(files)} images from '{class_name}' ...")
        for fname in files:
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).convert('L')        # grayscale
                img = img.resize(img_size, Image.LANCZOS)   # 32x32
                X.append(np.array(img, dtype=np.float32).flatten() / 255.0)
                y.append(label)
            except Exception as e:
                print(f"  [SKIP] {fpath}: {e}")

    return np.array(X), np.array(y)


print("=" * 55)
print("  Loading TRAINING data ...")
print("=" * 55)
X_train_full, y_train_full = load_images(TRAIN_DIR)
print(f"  Training samples: {len(X_train_full)}  "
      f"(indoor={np.sum(y_train_full==0)}, outdoor={np.sum(y_train_full==1)})")

print("\nLoading VALIDATION data ...")
X_val, y_val = load_images(VAL_DIR)
print(f"  Validation samples: {len(X_val)}  "
      f"(indoor={np.sum(y_val==0)}, outdoor={np.sum(y_val==1)})")


# ─────────────────────────────────────────────
#  2. HELPER: metrics + confusion matrix plot
# ─────────────────────────────────────────────
def evaluate(model, X, y, model_name, split_name="Validation"):
    preds = model.predict(X)
    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    cm   = confusion_matrix(y, preds)

    print(f"\n── {model_name} [{split_name}] ──")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['indoor', 'outdoor'])
    ax.set_yticklabels(['indoor', 'outdoor'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} — Confusion Matrix')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("/", "-")
    fig.savefig(os.path.join(RESULTS_DIR, f"cm_{safe_name}.png"), dpi=100)
    plt.close(fig)

    return {"model": model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


# ─────────────────────────────────────────────
#  3A. SUPERVISED DECISION TREE
#      Hyperparameter: max_depth = 5, 10, 20
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SUPERVISED DECISION TREE")
print("=" * 55)

dt_results = []
for depth in [5, 10, 20]:
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_split=10,
                                 criterion='gini', random_state=42)
    dt.fit(X_train_full, y_train_full)
    r = evaluate(dt, X_val, y_val, f"DT (depth={depth})")
    dt_results.append(r)

# Best DT by F1
best_dt_cfg = max(dt_results, key=lambda x: x['f1'])
print(f"\n  Best DT config: {best_dt_cfg['model']}  F1={best_dt_cfg['f1']:.4f}")


# ─────────────────────────────────────────────
#  3B. GRADIENT BOOSTING
#      Hyperparameter: n_estimators = 50, 100, 200
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  GRADIENT BOOSTING")
print("=" * 55)

gb_results = []
for n_est in [50, 100, 200]:
    gb = GradientBoostingClassifier(n_estimators=n_est, max_depth=3,
                                     learning_rate=0.1, random_state=42)
    gb.fit(X_train_full, y_train_full)
    r = evaluate(gb, X_val, y_val, f"GB (n_est={n_est})")
    gb_results.append(r)

best_gb_cfg = max(gb_results, key=lambda x: x['f1'])
print(f"\n  Best GB config: {best_gb_cfg['model']}  F1={best_gb_cfg['f1']:.4f}")


# ─────────────────────────────────────────────
#  3C. SEMI-SUPERVISED DECISION TREE
#      - 20% labelled, 80% unlabelled from training
#      - 5 iterations, confidence threshold 0.85/0.15
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SEMI-SUPERVISED DECISION TREE")
print("=" * 55)

LABELED_RATIO  = 0.20
CONFIDENCE_THR = 0.85   # prob >= 0.85 → outdoor,  <= 0.15 → indoor
N_ITER         = 5

# Split training into initial labelled / unlabelled
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X_train_full, y_train_full,
    train_size=LABELED_RATIO, stratify=y_train_full, random_state=42
)

print(f"  Initial labelled  : {len(X_labeled)}")
print(f"  Initial unlabelled: {len(X_unlabeled)}")

semi_iter_scores = []

for iteration in range(1, N_ITER + 1):
    # Train on currently labelled data
    semi_dt = DecisionTreeClassifier(max_depth=10, min_samples_split=10,
                                      criterion='gini', random_state=42)
    semi_dt.fit(X_labeled, y_labeled)

    val_metrics = evaluate(semi_dt, X_val, y_val,
                           f"Semi-DT iter={iteration}")
    semi_iter_scores.append(val_metrics['f1'])

    if len(X_unlabeled) == 0:
        print("  No unlabelled data left — stopping early.")
        break

    # Pseudo-label high-confidence unlabelled samples
    probs = semi_dt.predict_proba(X_unlabeled)[:, 1]  # P(outdoor)
    confident_mask = (probs >= CONFIDENCE_THR) | (probs <= (1 - CONFIDENCE_THR))
    pseudo_labels  = (probs >= CONFIDENCE_THR).astype(int)

    n_confident = confident_mask.sum()
    print(f"  iter {iteration}: {n_confident} high-confidence pseudo-labels added")

    if n_confident == 0:
        print("  No confident predictions — stopping early.")
        break

    # Add confident pseudo-labelled samples to labelled pool
    X_labeled   = np.vstack([X_labeled,   X_unlabeled[confident_mask]])
    y_labeled   = np.hstack([y_labeled,   pseudo_labels[confident_mask]])
    X_unlabeled = X_unlabeled[~confident_mask]
    y_unlabeled = y_unlabeled[~confident_mask]

    print(f"  Labelled pool size: {len(X_labeled)}  |  "
          f"Remaining unlabelled: {len(X_unlabeled)}")


# ─────────────────────────────────────────────
#  4. COMPARISON PLOT
# ─────────────────────────────────────────────
all_results = dt_results + gb_results
labels  = [r['model'] for r in all_results]
metrics = ['accuracy', 'precision', 'recall', 'f1']
colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

x = np.arange(len(labels))
width = 0.18

fig, ax = plt.subplots(figsize=(max(10, len(labels)*1.4), 5))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [r[metric] for r in all_results]
    ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — DT vs Gradient Boosting')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=120)
plt.close(fig)

# Semi-supervised F1 over iterations
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(range(1, len(semi_iter_scores) + 1), semi_iter_scores,
         marker='o', color='#4C72B0')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('F1-Score (Validation)')
ax2.set_title('Semi-Supervised DT — F1 over Iterations')
ax2.set_ylim(0, 1)
plt.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, "semi_supervised_f1.png"), dpi=120)
plt.close(fig2)

print("\n" + "=" * 55)
print("  ALL DONE!")
print(f"  Plots saved in: ./{RESULTS_DIR}/")
print("=" * 55)
