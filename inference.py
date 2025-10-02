from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import os


model_path = "./vit_model"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    label = model.config.id2label[pred]
    return label



test_image = "test.jpg"
predicted_class = predict_image(test_image)
print(f"Predicted class: {predicted_class}")



def predict_folder(folder_path):
    results = {}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, fname)
            label = predict_image(path)
            results[fname] = label
    return results