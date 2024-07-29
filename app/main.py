from pathlib import Path
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO 
from pydantic import BaseModel, EmailStr
from typing import List,Optional, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import io
import logging
import os

from utils.mongo_client import get_mongo_client
from utils.embedding import perform_vector_search
from utils.modelsHandler import predict
from utils.data_formater import formatter

from dotenv import load_dotenv

import json

from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
import hashlib
import uuid
from fastapi.responses import JSONResponse

#agent utils
from agent.graph_workflow_app import setup_lang_app
from pprint import pprint

# from ollama import generate
from langchain_community.chat_models import ChatOllama

#data analysis chain
from agent.chains.analyser import analyser
from agent.chains.generator_chain import generate_common
from agent.flow_state import set_analyser
from agent.flow_state import model
#analyser for diagnosis generation
analyser=set_analyser()
llm=model()
generate_common=generate_common(local_llm="llama3.1", llm=llm)


#inicio y cierre de app fast api

@asynccontextmanager
async def lifespan(app: FastAPI):
    client=await get_mongo_client()
    try:
        app.state.client = client
        yield
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
    finally:
        await client.close()  # Ensure the client is properly closed
        logger.info("MongoDB connection closed")


#fastapi app
app = FastAPI(lifespan=lifespan)

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
        if contents:
            print("************************got conteent*******************")
            img = Image.open(io.BytesIO(contents))
        else:
            print("****************************no content*****************************")

        # Middleware para formato compatible
        if img.format.lower() not in ["jpeg", "jpg", "png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")

        results = predict(img, x, y, width, height, model)

        logger.info("predict function finished")


        # Load mongo client
        DB_NAME = os.getenv("MONGO_DB_NAME")
        db = app.state.client[DB_NAME]
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
langgraph_app = setup_lang_app()

# User data model 
class activities(BaseModel):
    _id: str
    duration: int
    name: str
    MET:float
class DailyActivity(BaseModel):
    date: str
    activities: List[activities]
         
class Anthropometry(BaseModel):
    height: Optional[float] = 0.0
    weight: Optional[float] = 0.0
    BMI: Optional[float] = 0.0
    waist_circumference: Optional[float] = 0.0

class BiochemicalIndicators(BaseModel):
    glucose: Optional[str] = ""
    cholesterol: Optional[str] = ""

class Diet(BaseModel):
    ingest_preferences: Optional[List[str]] = []
    fruits_and_vegetables: Optional[str] = ""
    fiber: Optional[str] = ""
    saturated_fats: Optional[str] = ""
    sugars: Optional[str] = ""
    today_meals: Optional[List[str]] = []

class SocialIndicators(BaseModel):
    marital_status: Optional[str] = ""
    income: Optional[str] = ""
    access_to_healthy_foods: Optional[str] = ""

class Goals(BaseModel):
    Goals: Optional[List[str]] = []

class PotentialDiseases(BaseModel):
    PotentialDiseases: Optional[List[str]] = []

class Dietetics(BaseModel):
    age: Optional[int] = 0
    lifestyle: Optional[str] = ""
    job: Optional[str] = ""
    anthropometry: Optional[Anthropometry] = Anthropometry()
    biochemical_indicators: Optional[BiochemicalIndicators] = BiochemicalIndicators()
    diet: Optional[Diet] = Diet()
    social_indicators: Optional[SocialIndicators] = SocialIndicators()
    physical_activity: Optional[List[str]] = []
    daily_activities: Optional[List[DailyActivity]] = []
    activities_energy_consumption: Optional[int] = 0
    goals: Optional[List[str]] =[]
    potential_diseases: Optional[List[str]] =[]

class UserData(BaseModel):
    name: Optional[str] = "Djduddbdbh"
    lastName: Optional[str] = "Ejejdnrb"
    email: Optional[str] = "noprovided@email.com"
    sessionID: Optional[str] = ""
    dietetics: Optional[Dietetics] = Dietetics()

class InputModel(BaseModel):
    userData: Optional[UserData] = UserData()

class InputData(BaseModel):
    message: str


# ConnectionManager para gestionar sesiones
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_map: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        self.active_connections.append(websocket)
        self.session_map[websocket] = session_id

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if websocket in self.session_map:
            del self.session_map[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

async def generate_response(websocket: WebSocket, data: InputData, session_id: str, user_data:str):
    try:
        inputs = {"question": data.message, "sessionid": session_id, "user_data":user_data}
        response_data = []
        await manager.send_personal_message("START_OF_RESPONSE", websocket)
        async for event in langgraph_app.astream_events(input=inputs, version="v2"):
            kind = event["event"]
            print(kind)
            tags = event.get("tags", [])
            if kind == "on_chat_model_stream" and "tool_llm" in tags:
                data = event["data"]["chunk"].content
                if data:
                    # Empty content in the context of OpenAI or Anthropic usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    print(data, end="")
                if data not in response_data:
                    response_data.append(data)
                    # await manager.send_personal_message("".join(response_data), websocket)
                    print(data)
                    await manager.send_personal_message(data, websocket)
        print(response_data)
        await manager.send_personal_message("END_OF_RESPONSE", websocket)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await websocket.send_text(f"Error processing message: {str(e)}")

@app.websocket("/ws/langgraph/agent/")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        # Recibir el primer mensaje que contiene el userData
        initial_data = await websocket.receive_json()
        print(initial_data["message"])

        # Generar session_id basado en userData
        user_data = initial_data.get("userData", {})
        if user_data:
            session_id = user_data.get("id", "")
        else:
            session_id = str(uuid.uuid4())
        logger.info(f"Got: {session_id}")

        DB_NAME = os.getenv("MONGO_DB_NAME")
        db = app.state.client[DB_NAME]
        diagnosis_collection = db["diagnosis"]

        if session_id:
            print("tenemos session id")
        # Verificar si existe un diagnóstico en la colección
        existing_diagnosis = await diagnosis_collection.find_one({"session_id": session_id})
        print(existing_diagnosis)
        if existing_diagnosis:
            analysis = existing_diagnosis["analysis"]
            print("habia documento")
        else:
            # Ejecutar el análisis y almacenar el resultado en la colección
            print("puede que sea aqui")
            formated_data = formatter(user_data=UserData(**user_data))
            analysis = await analyser.ainvoke(input={formated_data})
            await diagnosis_collection.insert_one({"session_id": session_id, "analysis": analysis})
            print("diagnosis added")


        if initial_data["message"]:
            await manager.send_personal_message("START_OF_RESPONSE", websocket)
            async for chunk in generate_common.astream(input={"question":initial_data["message"]}, config={"configurable": {"session_id": session_id}}):
                await manager.send_personal_message(chunk, websocket)
            await manager.send_personal_message("END_OF_RESPONSE", websocket)

        await manager.connect(websocket, session_id)
        
        while True:
            data = await websocket.receive_json()
            print(data["message"])
            message = InputData(message=data["message"])
            # await websocket.send_text(f"Received message: {message.message}")
            # user_data_model = UserData(**user_data)
            await generate_response(websocket, message, user_data=analysis, session_id=session_id)
            # await websocket.send_text(f"Received message: {message.message}")
            # await websocket.send_text(f"Received user data: {user_data_model.email}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error receiving initial data: {str(e)}")
        await websocket.send_text(f"Error receiving initial data: {str(e)}")

# @app.websocket("/ws/langgraph/agent/")
# async def websocket_endpoint(websocket: WebSocket):
#     logger.info("Before accepting the connection")
#     await websocket.accept()
#     logger.info("Connection accepted")
#     try:
#         initial_data = await websocket.receive_json()
#         logger.info(f"Initial data received: {initial_data}")

#         user_data = initial_data.get("userData", {})
#         email = user_data.get("email", "")
#         logger.info(f"got email: {user_data}")

#         if email:
#             email_hash = hashlib.sha256(email.encode('utf-8')).hexdigest()
#             session_id = f"{email_hash}-{uuid.uuid4()}"
#         else:
#             session_id = str(uuid.uuid4())
#         logger.info(f"Got: {session_id}")

#         await manager.connect(websocket, session_id)
#         logger.info("Manager connected")

#         user_data_model = UserData(**user_data)
#         logger.info("User data model created")

#         while True:
#             data = await websocket.receive_json()
#             logger.info("Message received")
#             message = InputData(message=data["message"])
#             await generate_response(websocket, message, user_data_model, session_id)
#             logger.info(f"Response sent for message: {data['message']}")
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#         logger.info("WebSocket disconnected")
#     except Exception as e:
#         logger.error(f"Error receiving initial data: {str(e)}")
#         await websocket.send_text(f"Error receiving initial data: {str(e)}")

@app.get("/user/sessions/{email}")
async def get_user_sessions(email: str):
    logger.info(f"Fetching sessions for email: {email}")
    return manager.user_sessions.get(email, [])

@app.get("/session/messages/{session_id}")
async def get_session_messages(session_id: str):
    logger.info(f"Fetching messages for session ID: {session_id}")
    history = SQLChatMessageHistory(session_id, "sqlite:///memory.db")
    return history.get_messages()

@app.post("/langgraph/agent/")
async def process_message(data: InputData, userData: UserData = UserData()):
    logger.info(f"Processing message: {data.message} | User: {userData.email}")
    return StreamingResponse(generate_response(data=data, userData=userData), media_type="text/plain")
   

@app.post("/process-image/")
async def process_image_endpoint(image: UploadFile = File(...)):
    try:
        # Leer la imagen cargada
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convertir la imagen a bytes
        with io.BytesIO() as buffer:
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

        # Generar una descripción de la imagen
        full_response = ''
        for response in generate(
            model='llava:13b', 
            prompt="""Please give me the list of all food's ingredients that you see in the image in JSON format, I just want the list, do not tell me anything else 
            {"food_name": "food Name", "ingredients": ["ingrdient", "ingredient"]} provide the json structure with no premable or explanation and only return the raw json structure i gave you """, 
            images=[image_bytes], 
            stream=True,
            format='json',
        ):
            full_response += response['response']

        # Parsear la respuesta a formato JSON
        json_response = json.loads(full_response)

        return JSONResponse(content=json_response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)