from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from llama_cpp import Llama
from model.model import predict
from PIL import Image, UnidentifiedImageError
import io

app = FastAPI()

# Initialize the model globally and set it to None initially
llm = None
try:
    # Load the model once when the server starts
    llm = Llama(
        model_path="model/llama-3-8b-Instruct.Q2_K.gguf",
        chat_format="llama-2",
        n_gpu_layers=-1
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    # You might want to handle this differently depending on your application needs
    raise RuntimeError("Failed to load the model at startup")

@app.get("/")
def home():
    return {"Health": "OK"}

@app.post('/llama/')
async def prediction_llama(prompt: str):
    print('Predicting')
    try:
        if llm is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        output = llm(prompt, max_tokens=1024, echo=False)
        return output['choices'][0]['text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during llama prediction: {str(e)}")

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
