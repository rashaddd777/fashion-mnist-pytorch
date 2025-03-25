import os
from torchvision import datasets

os.makedirs("images", exist_ok=True)
test_dataset = datasets.FashionMNIST(root="data", train=False, download=True)
for idx in range(10):  # change 10 to any number of images you want to save
    img, label = test_dataset[idx]
    img.save(os.path.join("images", f"fashionmnist_test_{idx}.png"))
