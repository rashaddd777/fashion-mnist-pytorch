import os
import requests

url = "https://fashionmnist-api.herokuapp.com/predict"
images_dir = "images"

for image_file in os.listdir(images_dir):
    if image_file.endswith(".png"):
        image_path = os.path.join(images_dir, image_file)
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print(f"{image_file}: {response.json()}")
            else:
                print(f"Error for {image_file}: {response.text}")
