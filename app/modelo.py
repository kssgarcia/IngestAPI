
import torch
import torchvision
from typing import List
from PIL import Image
from io import BytesIO
import yaml
import logging
from bson import ObjectId
from pathlib import Path
import os
from fastapi import UploadFile, HTTPException, File
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the response model
class Prediction(BaseModel):
    id: str
    Clase: str
    probabilidad: float

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

# Open the config file using a context manager
ruta_config=os.path.join(os.path.dirname(__file__), "config.yaml")
try:

    with open(ruta_config) as f:
        config = yaml.safe_load(f)
        logger.info(f"se establecion configuarcion desde .config")
except Exception as e:
    logger.info(f"algo pasa con la ruta {ruta_config}. {str(e)}")

# Assign values from the dictionary to local variables
clases = config.get("class_names", [])
device = config.get("device", "cpu")
ruta_modelo = config.get("model_path", "mi_modelo.pth")
parametros_Transform = config.get("transform_params", {})

#Los directorios de Docker pueden llegar a variar por lo que me asegura de que la ruta sea la correcta
BASEDIR= Path(__file__).resolve(strict=True).parent
RUTA=f"{BASEDIR}/{ruta_modelo}"

# Load the model
model = torchvision.models.vit_b_16(weights=None).to(device)
try:
    model.load_state_dict(torch.load(RUTA, map_location=torch.device(device)), strict=False)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")

# Set the model to evaluation mode
model.eval()

transform =  torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(parametros_Transform.get("resize", 224), antialias=True),
    torchvision.transforms.CenterCrop(parametros_Transform.get("center_crop", 224)),
    torchvision.transforms.Normalize(
        parametros_Transform.get("mean", [0.485, 0.456, 0.406]),
        parametros_Transform.get("std", [0.229, 0.224, 0.225])
    ),
])

# Define the transform function
def transform_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_top_predictions(image_tensor, top_k=2):
    output = model(image_tensor)
    output = torch.softmax(output, dim=1)

    # Get the top k predictions along with their probabilities
    probabilities, indices = torch.topk(output, top_k, dim=1)

    # Convert indices to class names
    class_ids = [str(idx.item()) for idx in indices[0]]
    class_names = [clases[idx.item()] for idx in indices[0]]

    # Create a list of Pydantic model instances
    predictions = [
        Prediction(id=id, Clase=name, probabilidad=prob.item())
        for id, name, prob in zip(class_ids, class_names, probabilities[0])
    ]

    return PredictionResponse(predictions=predictions)





