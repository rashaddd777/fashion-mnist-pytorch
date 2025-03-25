import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ─── Import your model class ────────────────────────────────────────────────────
from model import EnhancedCNN

# ─── Config ─────────────────────────────────────────────────────────────────────
IMAGES_DIR = "images"
CHECKPOINT = "model_checkpoint.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "T‑shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ─── Preprocessing ──────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ─── Load Model ─────────────────────────────────────────────────────────────────
model = EnhancedCNN().to(DEVICE)
model = torch.jit.load(CHECKPOINT, map_location=DEVICE)
model.to(DEVICE).eval()

# ─── Inference Loop ─────────────────────────────────────────────────────────────
for filename in sorted(os.listdir(IMAGES_DIR)):
    if not filename.lower().endswith((".png", ".jpg")):
        continue

    path = os.path.join(IMAGES_DIR, filename)
    img = Image.open(path)
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    label = CLASS_NAMES[pred]

    print(f"{filename: <25} → {label}")

    # Optional: display image with title
    plt.imshow(img, cmap="gray")
    plt.title(label)
    plt.axis("off")
    plt.show()
