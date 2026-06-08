import uvicorn
from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
import cv2
import numpy as np

app = FastAPI()

model_new = keras.models.load_model('mnist.hdf5')


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    contents = await img.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    pred = model_new.predict(img_grey.reshape(1, 28, 28, 1))
    return {"result": float(np.argmax(pred[0])), "percent": pred[0].tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
