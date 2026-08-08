from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from pathlib import Path
import streamlit as st
from PIL import Image
import torch.nn as nn
import torch
import io


BASE_DIR = Path(__file__).parent

# ── Model ──────────────────────────────────────────────────────────────────────
class TrafficSignClassifier(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── Transform ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load model ─────────────────────────────────────────────────────────────────
model = TrafficSignClassifier(num_classes=43)

torch.load(BASE_DIR / 'model_PDD_TrafficSignClassifier.pth', map_location=device)
labels = torch.load(BASE_DIR / 'labels_PDD_TrafficSignClassifier.pth')

model.to(device)
model.eval()

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI()

@app.post('/predict')
async def predict(file: UploadFile = File()):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(400, detail='Файл кошулган жок')

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()
            sign_name = labels[prediction] if isinstance(labels, list) else str(prediction)

        return {
            'class_id': prediction,
            'sign': sign_name
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)


# ── Streamlit ──────────────────────────────────────────────────────────────────
st.title('PDD Traffic Sign Classifier')
st.text('Загрузите изображение дорожного знака, и модель попробует его распознать.')
uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])
if not uploaded_file:
    st.info('Загрузите изображение')
else:
    st.image(uploaded_file, caption='Загруженное изображение')
    if st.button('Распознать знак'):
        try:
            image        = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                y_prediction = model(image_tensor)
                prediction   = y_prediction.argmax(dim=1).item()
            sign_name = labels[prediction] if isinstance(labels, list) else str(prediction)
            st.success(f'Модель думает, что это: {sign_name}')
        except Exception as e:
            st.error(f'Ошибка: {str(e)}')
