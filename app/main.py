from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from model.model import predict
from PIL import Image, UnidentifiedImageError
import io

app = FastAPI()

@app.get("/")
def home():
    return {"Health": "OK"}

@app.post('/file/')
async def prediction(file: UploadFile = File(...), x: float = Form(0), y: float = Form(0), width: float = Form(0), height: float = Form(0)):
    # Check the file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        # Read the contents of the uploaded file
        contents = await file.read()
        # Convert the contents to a PIL Image
        img = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot identify image file. Ensure the file is a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")
    try:
        results = predict(img, x, y, width, height)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    
    return results

