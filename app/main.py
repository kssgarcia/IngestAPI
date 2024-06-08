from pathlib import Path
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
import io
import logging
import os
from utils.mongo_client import get_mongo_client
from utils.embedding import perform_vector_search
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve(strict=True).parent

class Prediction(BaseModel):
    prediction: str
    prob: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

class foodDocument(BaseModel):
    nombre: str
    ingredientes: List[str]

class foodResponse(BaseModel):
    food: List[List[foodDocument]]

class PredictionWithFoodsResponse(BaseModel):
    predictions: List[Prediction]
    foods: List[List[foodDocument]]

# Inicializa el modelo una vez
model = YOLO('models/scrap100.pt')

def predict(img: Image, x: float, y: float, width: float, height: float, model: YOLO):
    left = x - width / 2
    top = y - height / 2
    if x + y + width + height > 0:
        image = crop(img, top=int(top), left=int(left), height=int(height), width=int(width))
    else:
        transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize the image
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
        image = transform(img).unsqueeze(0)
    result = model(image)
    names = result[0].names
    probs = result[0].probs.top5
    conf = result[0].probs.top5conf.tolist()
    response = [Prediction(prediction=names[key], prob=conf[i]) for i, key in enumerate(probs)]
    return PredictionResponse(predictions=response)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Carga las variables de entorno desde .env

app = FastAPI()

class mlPrompt(BaseModel):
    usermessage: str

class mlresponse(BaseModel):
    sytemmessage: str

# Modelo de solicitud para el endpoint /vs
class VectorSearchRequest(BaseModel):
    query: str
    index: str  # Índice de búsqueda en la base de datos

# Modelo de respuesta para el endpoint /vs
class VectorSearchResponse(BaseModel):
    results: List[str]

# Dependencia para pasar el modelo de YOLO
def get_yolo_model():
    return model

@app.get("/")
def home():
    return {"Health": "OK"}

@app.post('/file/')
async def prediction(
    file: UploadFile = File(...), 
    x: float = Form(0), 
    y: float = Form(0), 
    width: float = Form(0), 
    height: float = Form(0), 
    model: YOLO = Depends(get_yolo_model),
):
    logger.info("Starting the predict function")
    try:
        contents = await file.read()
        
        # Verificación del contenido
        if not contents:
            raise HTTPException(status_code=400, detail="The file is empty")

        img = Image.open(io.BytesIO(contents))

        # Middleware para formato compatible
        if img.format.lower() not in ["jpeg", "jpg", "png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")

        results = predict(img, x, y, width, height, model)

        # Load mongo client
        client = await get_mongo_client()
        DB_NAME = os.getenv("MONGO_DB_NAME")
        db = client[DB_NAME]
        collection = db.platos

        # Insertar las predicciones en MongoDB
        db.predictions.insert_one(results.dict())
        logger.info("Finished the predict function")

        # Búsqueda vectorial
        foods = []  # 5 listas de 3 documentos
        pred = results.dict().get("predictions")
        foods.append(await perform_vector_search(query=pred[0]["prediction"], index="nombreEmbedding"))
        logger.info("sacando response")
        
        # Retorna las predicciones y los resultados de la búsqueda vectorial
        return {"predictions": pred, "Plate": foods}
    except HTTPException as e:
        logger.error(f"HTTPException in predict function: {str(e.detail)}")
        raise
    except Exception as e:
        logger.error(f"Error in predict function: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# Endpoint para la búsqueda vectorial
@app.post("/vs", response_model=VectorSearchResponse)
async def vector_search(req: VectorSearchRequest):
    try:
        results = await perform_vector_search(req.query, req.index)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para la búsqueda vectorial
@app.post("/nlp", response_model=mlresponse)
async def NLP(req: mlPrompt):
    try:
        content = req
        # Example: reuse your existing OpenAI setup
        from openai import OpenAI

        # Point to the local server
        client = OpenAI(base_url="https://lznfhzqb-1233.use2.devtunnels.ms/v1", api_key="lm-studio")

        completion = client.chat.completions.create(
            model="model-identifier",
            messages=[
                {"role": "system", "content": "Always answer in rhymes."},
                {"role": "user", "content": content.dict()["usermessage"]}
            ],
            temperature=0.7,
        )
        logger.info("Retornanado")
        return mlresponse(sytemmessage=completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))