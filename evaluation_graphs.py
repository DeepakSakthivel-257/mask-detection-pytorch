import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# ----------------------------
# CONFIGURATION
# ----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = "models/mask_detector.pth"
data_dir = "data/split/test"

# ----------------------------
# DATA LOADING
# ----------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = test_dataset.classes
print("Classes:", classes)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# EVALUATION
# ----------------------------
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 0]

        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# ----------------------------
# METRICS
# ----------------------------
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"\nModel Performance:\nAccuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# ----------------------------
# ROC CURVE
# ----------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.show()

# ----------------------------
# PRECISION–RECALL CURVE
# ----------------------------
precision, recall, _ = precision_recall_curve(all_labels, all_probs)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.tight_layout()
plt.savefig("results/precision_recall_curve.png")
plt.show()

# ----------------------------
# METRIC SUMMARY BAR CHART
# ----------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [acc, prec, rec, f1]

plt.figure(figsize=(7, 4))
bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'violet'])
plt.ylim(0, 1.1)
plt.title('Model Performance Summary')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.1, yval + 0.02, f"{yval:.2f}")
plt.tight_layout()
plt.savefig("results/metric_summary.png")
plt.show()

# ----------------------------
# TRAINING HISTORY (if available)
# ----------------------------
try:
    history = torch.load("models/training_history.pth")

    plt.figure(figsize=(6, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/accuracy_curve.png")
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss_curve.png")
    plt.show()
except:
    print("⚠️ No training history found — skipping accuracy/loss plots.")
