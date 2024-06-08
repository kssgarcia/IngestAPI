import yaml
import logging
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()  # Carga las variables de entorno desde .env

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MongoDB connection setup
async def get_mongo_client():

    # Obtener la cadena de conexión desde una variable de entorno
    MONGO_USERNAME = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
  

    # Obtener la cadena de conexión desde una variable de entorno
    MONGO_URI = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.q4lvimh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


    # Crear el cliente de forma asíncrona
    client = MongoClient(MONGO_URI,server_api=ServerApi('1'))
    try:
      client.admin.command('ping')
      logger.info(f"Se obtuvo cliente {client}")
    except Exception as e:
      logger.info(f"hubo un error en la obtencion del cliente{client}: {e}")

    return client



