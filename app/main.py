from pathlib import Path
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from pydantic import BaseModel, EmailStr
from typing import List,Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
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

#user data model 

class Anthropometry(BaseModel):
    height: float= 0.0
    weight: float = 0.0
    BMI: float = 0.0
    waist_circumference: float = 0.0

class BiochemicalIndicators(BaseModel):
    glucose: str = ""
    cholesterol: str = ""

class Diet(BaseModel):
    ingest_preferences: List[str]= []
    fruits_and_vegetables: str = ""
    fiber: str = ""
    saturated_fats: str = ""
    sugars: str = ""
    today_meals: List[str] =[]

class SocialIndicators(BaseModel):
    marital_status: str = ""
    income: str = ""
    access_to_healthy_foods: bool = False

class Goals(BaseModel):
    reduce_weight: bool = False
    reduce_waist_circumference: bool = False
    improve_glucose_cholesterol_levels: bool = False
    increase_intake_fruits_vegetables_fiber: bool = False
    reduce_intake_processed_food_saturated_fats_sugars: bool= False
    increase_physical_activity: bool = False
    improve_cardio_health: bool = False
    reduce_diabetes_risks: bool = False

class PotentialDiseases(BaseModel):
    type_2_diabetes: bool = False
    heart_diseases: bool = False
    hypertension: bool = False
    overweight_obesity: bool = False

class Dietetics(BaseModel):
    age: int = 0
    lifestyle: str = ""
    job: str = ""
    anthropometry: Anthropometry
    biochemical_indicators: BiochemicalIndicators
    diet: Diet
    social_indicators: SocialIndicators
    physical_activity: str = ""
    daily_activities: List[str] = []
    activities_energy_consumption: int = 0
    goals: Goals
    potential_diseases: PotentialDiseases

class UserData(BaseModel):
    email: str = "noprovided@email.com"

class InputModel(BaseModel):
    userData: UserData
    dietetics: Dietetics



#Input question model
class InputData(BaseModel):
    message: str


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

async def generate_response(websocket: WebSocket, data: InputData, userData: InputModel):
    try:
        thread = {"configurable": {"thread_id": "4"}}
        inputs = {"question": data.message, "userdata": userData}
        response_data = []
        for event in langgraph_app.stream(inputs, stream_mode="values"):
            for key, value in event.items():
                if key == "generation":
                    response_data.append(value)
                    await manager.send_personal_message("".join(response_data), websocket)
        await manager.send_personal_message("END_OF_RESPONSE", websocket)
    except Exception as e:
        await websocket.send_text(f"Error processing message: {str(e)}")
    except Exception as e:
        await websocket.send_text(f"Error processing message: {str(e)}")

@app.websocket("/ws/langgraph/agent/")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            print(data["message"])
            message = InputData(message=data["message"])
            user_data = InputModel(**data["userData"])
            await generate_response(websocket, message, user_data)
            await websocket.send_text(f"Received message: {message.message}")
            await websocket.send_text(f"Received user data: {user_data.userData.email}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/langgraph/agent/")
async def process_message(data: InputData, userData: InputModel):
    return StreamingResponse(generate_response(data=data, userData=userData), media_type="text/plain")
    
   
