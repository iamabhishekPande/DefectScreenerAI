
import os
import time
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Paths
data_dir = "/workspace/finaldataset"
model_path = "/workspace/resnet50_final.pt"
output_dir = "/workspace/model"
os.makedirs(output_dir, exist_ok=True)

# Load the model
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Class names (assuming alphabetical order: Defective, Non-defective)
class_names = ['Defective', 'Non-defective']

# Transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load test data
test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Evaluation
all_preds = []
all_labels = []
false_pos = []
false_neg = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i in range(inputs.size(0)):
            true = labels[i].item()
            pred = preds[i].item()
            if true != pred:
                if pred == 1:
                    false_pos.append((inputs[i].cpu(), true, pred))
                else:
                    false_neg.append((inputs[i].cpu(), true, pred))

# Metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

# Save metrics
with open(os.path.join(output_dir, "metrics_report.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {prec:.4f}\n")
    f.write(f"Recall: {rec:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# Error visualization
def save_errors(errors, title, filename):
    if len(errors) == 0:
        print(f"⚠️ No {title.lower()} to display.")
        return
    fig, axes = plt.subplots(1, len(errors), figsize=(15, 5))
    if len(errors) == 1:
        axes = [axes]
    for i, (img, label, pred) in enumerate(errors):
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'True: {class_names[label]}\nPred: {class_names[pred]}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

save_errors(false_pos[:5], "False Positives", "false_positives.png")
save_errors(false_neg[:5], "False Negatives", "false_negatives.png")

# Inference speed
sample_size = min(100, len(test_data))
sample_inputs = torch.stack([test_data[i][0] for i in range(sample_size)])
sample_inputs = sample_inputs.to(device)

start_time = time.time()
with torch.no_grad():
    _ = model(sample_inputs)
end_time = time.time()

avg_infer_time = (end_time - start_time) / sample_size
with open(os.path.join(output_dir, "metrics_report.txt"), "a") as f:
    f.write(f"\nAverage Inference Time (CPU): {avg_infer_time:.6f} seconds/image\n")
    f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters())}\n")
    f.write(f"Model File Size: {os.path.getsize(model_path) / 1e6:.2f} MB\n")

print("✅ Evaluation complete. All outputs saved to:", output_dir)

