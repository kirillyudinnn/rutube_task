import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
from model import get_food101_model
import uvicorn

app = FastAPI(
    title='Food101_Classifier',
    version='0.1.0'
)


@app.post("/classify")
async def classify_image(images: List[UploadFile] = File(...)):
    results = []

    for image in images:
        with Image.open(image.file) as img:
            img = img.convert('RGB')
            model = get_food101_model()
            img_tensor = torch.unsqueeze(model.transform(img), 0)
            predict = model.predict(img_tensor)[0]
            results.append({"filename": 'image', "prediction": predict})
            return JSONResponse(content=results)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)