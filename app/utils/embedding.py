from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from utils.mongo_client import get_mongo_client
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import os
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#prueba de busqueda
# Modelo de solicitud para el endpoint /vs
class VectorSearchRequest(BaseModel):
    query: str
    index: str  # Índice de búsqueda en la base de datos

# Modelo de respuesta para el endpoint /vs
class VectorSearchResponse(BaseModel):
    results: List[str]

class foodDocument(BaseModel):
    name:str
    ingredients: List[str]

class foodResponse(BaseModel):
    food: List[foodDocument]

#funcion para realizar embedding de busqueda
def generateEmbedding(text: str) -> list[float]:

    import requests
    #jina_958f2f6f4fbb463b80367d493c9a875fRRylBWRxDmvDR6OceXRUDIwKmsNS
    url = 'https://api.jina.ai/v1/embeddings'

    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer jina_958f2f6f4fbb463b80367d493c9a875fRRylBWRxDmvDR6OceXRUDIwKmsNS'
    }

    data = {
    'input': [text],
    'model': 'jina-embeddings-v2-base-es'
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    
    return response.json()["data"][0]["embedding"]


#Se crean funciones para optimizacion de uso de modelo de embedding.
async def get_embedding_from_db(query: str, db: MongoClient) -> list[float]:
    logger.info("buscando")
    collection = db['embeddings']
    result = collection.find_one({"query": query})
    if result:
        return result["embedding"]
    else:
        return None

async def save_embedding_to_db(query: str, embedding: list[float], db: MongoClient):
    logger.info("guardando")
    collection = db['embeddings']
    collection.insert_one({"query": query, "embedding": embedding})

# Función para realizar la búsqueda vectorial en la base de datos
async def perform_vector_search(query: str, index: str) -> List[str]:
    client = await get_mongo_client()
    DB_NAME = os.getenv("MONGO_DB_NAME")
    db = client[DB_NAME]
    
    logger.info("a ver si existe")
    existing_embedding = await get_embedding_from_db(query, db)
    if existing_embedding:
        # Si el embedding ya existe en la base de datos, no es necesario generar uno nuevo
        embedding = existing_embedding
        logger.info("sí habia uno")
    else:
        # Si el embedding no existe, genera uno nuevo
        logger.info("No habia así que voy a generar")
        embedding = generateEmbedding(query)
        # Guarda el nuevo embedding en la base de datos para futuras consultas
        logger.info("ahora lo guardo")
        await save_embedding_to_db(query, embedding, db)

    logger.info("a que vector voy")
    collection_name = None
    if index == "nombreIngreEmbedding":
        collection_name = "ingredientes"
    elif index == "nombreEmbedding":
        collection_name = "platos"
    else:
        raise HTTPException(status_code=500, detail="El índice no es válido")
    
    
    logger.info("ejecutando busqueda")
    collection = db[collection_name]
    results = collection.aggregate([
        {"$vectorSearch": {
            "queryVector": generateEmbedding(query),
            "path": index,
            "numCandidates": 10,
            "limit": 1,
            "index": index,
        }}
    ])

    logger.info("Busqueda terminada")
    listrsul=list(results)
    response_items = None
    if "name" in listrsul[0] and "ingredients" in listrsul[0]:
        response_items=foodDocument(name=listrsul[0]["name"], ingredients=listrsul[0]["ingredients"])
    else:
        logger.warning(f"Documento faltante de campos requeridos: {results[0]}")
    
    logger.info("mostrando resultados de busqueda")
    print(response_items)
    return response_items