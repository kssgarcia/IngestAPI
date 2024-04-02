# %%
from pathlib import Path
from torchvision.transforms.functional import crop
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve(strict=True).parent

def predict(img: Image, x: float, y: float, width: float, height: float):
    model = YOLO('model/scrap100.pt')
    # transform = transforms.Compose([
    #     transforms.Resize((640,640)),  # Resize the image
    #     transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    # ])
    left = x - width / 2
    top = y - height / 2
    cropped_image = crop(img, top=int(top), left=int(left), height=int(height), width=int(width))
    # image = transform(img).unsqueeze(0)  
    result = model(cropped_image)
    names = result[0].names
    probs = result[0].probs.top5
    conf = result[0].probs.top5conf.tolist()
    return [{'prediction': names[key], 'prob': conf[i]} for i,key in enumerate(probs)]