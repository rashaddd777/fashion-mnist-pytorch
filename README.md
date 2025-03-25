# Fashion‑MNIST PyTorch Classifier

A simple convolutional neural network (CNN) implemented in PyTorch that trains on the Fashion‑MNIST dataset and achieves **93.69% test accuracy** after 40 epochs. This repo includes scripts for data download, model training, evaluation, and inference on custom images.

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Download & Preprocessing](#data-download--preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference on Custom Images](#inference-on-custom-images)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [License](#license)

---

## 🚀 Project Structure

```
├── data/                     # Raw and processed dataset files
├── images/                   # Custom images for inference
├── model.py                  # Defines EnhancedCNN architecture
├── image_test.py             # Runs inference on images/
├── data_download.ipynb       # Notebook: download Fashion-MNIST
├── data_load.ipynb           # Notebook: preprocessing & dataloaders
├── test.py                   # Evaluate model on test set
├── app.py                    # Interactive inference CLI
├── model_checkpoint.pt       # TorchScript checkpoint (excluded from git)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/rashaddd777/fashion-mnist-pytorch.git
   cd fashion-mnist-pytorch
   ```
2. Create & activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Data Download & Preprocessing

Run the Jupyter notebook to automatically download Fashion‑MNIST raw files and convert them into PyTorch datasets:
```bash
jupyter notebook data_download.ipynb
data_load.ipynb
```

---

## 🏋️ Training

```bash
python train.py \
  --batch_size 64 \
  --epochs 40 \
  --lr 0.001
```
Checkpoint will be saved as `model_checkpoint.pt`.

---

## 📊 Evaluation

Evaluate test accuracy:
```bash
python test.py --checkpoint model_checkpoint.pt
```

---

## 🖼️ Inference on Custom Images

Place your `.png`/`.jpg` images in `images/`, then run:
```bash
python image_test.py
```
Each image’s predicted label will print and display.

---

## 📈 Results

| Epochs | Test Accuracy |
|--------|---------------|
| 40     | **93.69%**    |

---

## 🏗️ Model Architecture

`EnhancedCNN` consists of three convolutional blocks (Conv→BatchNorm→ReLU→MaxPool) followed by two fully-connected layers with dropout.

---

## 📦 Dependencies

See `requirements.txt` — tested on PyTorch 2.0+, torchvision, matplotlib.

---

## 📄 License

This project is licensed under the **MIT License** — see `LICENSE` for details.

