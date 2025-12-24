from PIL import Image
import torch
from torchvision import transforms


# ---- Classes (same order as training) ----
CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    tensor = _preprocess(image)
    return tensor.unsqueeze(0)


def predict(model, image_tensor: torch.Tensor, device: torch.device):
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return {
        "prediction": CLASSES[pred.item()],
        "confidence": round(conf.item(), 4),
        "class_index": pred.item(),
    }
