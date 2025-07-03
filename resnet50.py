
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Dataset directories
base_dir = '/workspace/finaldataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')
model_dir = "/workspace/model"
os.makedirs(model_dir, exist_ok=True)

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_dir, transform=val_test_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)


# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained ResNet-50
model = models.resnet50(pretrained=True)

# Freeze first 80% of layers
total_params = list(model.parameters())
freeze_until = int(len(total_params) * 0.8)
for i, param in enumerate(total_params):
    if i < freeze_until:
        param.requires_grad = False

# Replace final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: defective, non-defective
model = model.to(device)

# Loss and optimizer for frozen phase
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 🔒 Phase 1: Train with frozen layers (10 epochs)
print("🔒 Phase 1: Training with frozen layers (10 epochs)")
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    
    print(f"Train Loss: {running_loss:.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

# 🔓 Phase 2: Fine-tune all layers (40 epochs)
print("🔓 Phase 2: Fine-tuning all layers (40 epochs)")

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(40):
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/40"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

    print(f"[{epoch+1}/40] Train Loss: {total_loss:.4f} | Val Loss: {val_loss / len(val_loader):.4f}")
    scheduler.step()

# Save model
torch.save(model.state_dict(), "resnet50_final.pt")
print("Model training complete. Saved as 'resnet50_final.pt'")

