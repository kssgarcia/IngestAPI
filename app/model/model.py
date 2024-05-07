# %%
from pathlib import Path
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

import json
from llama_cpp import Llama

BASE_DIR = Path(__file__).resolve(strict=True).parent

def predict(img: Image, x: float, y: float, width: float, height: float):
    """
    Predict objects in the image using YOLO model with optional cropping.

    Parameters:
        img (PIL.Image): The image to process.
        x (float): X coordinate of the center of crop box.
        y (float): Y coordinate of the center of crop box.
        width (float): Width of the crop box.
        height (float): Height of the crop box.

    Returns:
        list: A list of dictionaries with prediction and probability.
    """
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
        
        model = YOLO('model/scrap100.pt')
        result = model(image)
        
        # Assuming the result structure is as expected
        names = result[0].names
        probs = result[0].probs.top5
        conf = result[0].probs.top5conf.tolist()
        return [{'prediction': names[key], 'prob': conf[i]} for i,key in enumerate(probs)]

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return [{"prediction": "Error processing the image", "prob": 0}]

def llama_predict(prompt: str):
    try:
        llm = Llama(
            model_path="model/llama-3-8b-Instruct.Q2_K.gguf",
            chat_format="llama-2"
        )
        prompt = '''
        Why is the sky blue?
        '''
        output = llm(prompt, max_tokens=5120, echo=False)
        return json.dumps(output, indent=2)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return [{"prediction": "Error processing the image", "prob": 0}]
