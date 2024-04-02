from fastapi import FastAPI, File, UploadFile, Form
from model.model import predict
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def home():
    return {"Health": "OK"}

@app.post('/file/')
async def prediction(file: UploadFile = File(...), x: float = Form(...), y: float = Form(...), width: float = Form(...), height: float = Form(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    results = predict(img, x, y, width, height)
    return results
