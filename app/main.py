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
from utils.modelsHandler import predict

#agent utils
from agent import graph_workflow_app
from pprint import pprint



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

        logger.info("predict function finished")


        # Load mongo client
        client = await get_mongo_client()
        DB_NAME = os.getenv("MONGO_DB_NAME")
        db = client[DB_NAME]
        collection = db.platos

        # Insertar las predicciones en MongoDB
        db.predictions.insert_one(results.dict())
        logger.info("Finished the predict function")

        # Búsqueda vectorial
        
        # foods = []  # 5 listas de 3 documentos
        
        # foods.append(await perform_vector_search(query=pred[0]["prediction"], index="nombreEmbedding"))
        pred = results.dict().get("predictions")
        logger.info("sacando response")
        
        # Retorna las predicciones y los resultados de la búsqueda vectorial
        return {"predictions": pred}
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



# Agent Endponit
langgraph_app = graph_workflow_app.setup_lang_app()

class InputData(BaseModel):
    message: str


@app.post("/langgraph/agent/")
async def process_message(data: InputData):
    try:
        thread = {"configurable": {"thread_id": "4"}}
        result = []
        inputs = {"question": data.message}
        for event in langgraph_app.stream(inputs, stream_mode="values"):
            print(event)
            for key, value in event.items():
                # Node
                print(value)
                pprint(f"Node '{key}':")
                
                # pprint(value["keys"], indent=2, width=80, depth=None)
            pprint("\n---\n")
        return(event["generation"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
    
   
