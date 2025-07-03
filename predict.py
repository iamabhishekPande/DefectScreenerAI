import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os

# Paths
model_path = "/workspace/resnet50_final.pt"
folder_path = "/workspace/finalTestImages"  # 🔁 Change this to your folder with images
output_file = "/workspace/model/predictions.txt"

# Load model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Class names
class_names = ['Defective', 'Non-defective']

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# Prediction loop
with open(output_file, 'w') as f:
    for filename in os.listdir(folder_path):
        if not any(filename.lower().endswith(ext) for ext in image_extensions):
            continue

        image_path = os.path.join(folder_path, filename)
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                label = class_names[pred.item()]
                print(f"{filename}: {label}")
                f.write(f"{filename}: {label}\n")

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

print(f"\nPredictions saved to: {output_file}")
