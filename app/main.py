from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.datastructures import UploadFile
from pydantic import BaseModel, Field
from llama_cpp import Llama
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from torchvision.transforms.functional import crop
from torchvision import transforms
from ultralytics import YOLO
import io

app = FastAPI()

BASE_DIR = Path(__file__).resolve(strict=True).parent

# Initialize the model globally and set it to None initially
llm = None

class LlamaPredictionRequest(BaseModel):
    prompt: str

# @app.on_event("startup")
# async def load_model():
#     global llm
#     try:
#         # Load the model once when the server starts
#         llm = Llama(
#             model_path="model/Phi-3-mini-128k-instruct.IQ1_S.gguf",
#             chat_format="llama-2",
#             n_gpu_layers=-1
#         )
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Failed to load model: {str(e)}")
#         raise RuntimeError("Failed to load the model at startup")

@app.get("/")
def home():
    return {"Health": "OK"}

@app.post('/llama/')
async def llama_prediction(request: LlamaPredictionRequest):
    print('Predicting')
    try:
        if llm is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        # output = llm(request.prompt, max_tokens=1024, echo=False)
        # return output['choices'][0]['text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during llama prediction: {str(e)}")

@app.post('/file/')
async def food_prediction(file: UploadFile = File(...), x: float = Form(0), y: float = Form(0), width: float = Form(0), height: float = Form(0)):
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
        if x + y + width + height > 0:
            left = int(x - width / 2)
            top = int(y - height / 2)
            image = crop(img, top=top, left=left, height=int(height), width=int(width))
            image = transforms.ToTensor()(image).unsqueeze(0)  # Convert to tensor
        else:
            transform = transforms.Compose([
                transforms.Resize((640, 640)),  # Resize the image
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            ])
            image = transform(img).unsqueeze(0)
        
        model = YOLO('models/scrap100.pt')
        result = model(image)
        
        # Assuming the result structure is as expected
        names = result[0].names
        probs = result[0].probs.top5
        conf = result[0].probs.top5conf.tolist()
        return [{'prediction': names[key], 'prob': conf[i]} for i,key in enumerate(probs)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
