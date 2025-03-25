from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torch.jit.load("model_checkpoint.pt", map_location=device)
model.eval()

inference_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")
        img = inference_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
        return {"prediction": int(pred.item())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
