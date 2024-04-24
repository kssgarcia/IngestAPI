from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from model.model import predict
from PIL import Image
import io
import logging
import os
from mongo_client import get_mongo_client
from embedding import perform_vector_search
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Carga las variables de entorno desde .env

app = FastAPI()

class Prediction(BaseModel):
    prediction: str
    prob: str

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

class foodDocument(BaseModel):
    nombre:str
    ingredientes: List[str]
class foodResponse(BaseModel):
    food: List[List[foodDocument]]

# Modelo de solicitud para el endpoint /vs
class VectorSearchRequest(BaseModel):
    query: str
    index: str  # Índice de búsqueda en la base de datos


# Modelo de respuesta para el endpoint /vs
class VectorSearchResponse(BaseModel):
    results: List[str]

@app.get("/")
def home():
    return {"Health": "OK"}

@app.post('/file/')
async def prediction(file: UploadFile = File(...), x: float = Form(0), y: float = Form(0), width: float = Form(0), height: float = Form(0)):
    logger.info("Starting the predict function")
    try:
        contents = await file.read()

        #Content verifying middleware
        if not contents:
            raise HTTPException(status_code=400, detail="The file is empty")

        img = Image.open(io.BytesIO(contents))

        #Compatible format middleware
        if img.format.lower() not in ["jpeg", "jpg", "png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")

        results = predict(img, x, y, width, height)

        #Load mongo client
        client = await get_mongo_client()
        DB_NAME = os.getenv("MONGO_DB_NAME")
        db = client[DB_NAME]
        collection=db.platos

        # Insert predictions as a single document into MongoDB
        db.predictions.insert_one(results.dict())
        print(results.dict())
        logger.info("Finished the predict function")

        #Vector search
        foods=[] #5 listas de 3 documentos
        for prediction in results.dict().get("predictions"):
            foods.append(await perform_vector_search(query=prediction["prediction"], index="nombreEmbedding"))
        print(foods)
        logger.info("sacando response")
        return foods
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