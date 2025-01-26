from fastapi import FastAPI, HTTPException
import asyncio
import pickle
import numpy
from classifier import trainingModel

app = FastAPI()

@app.get("/greeting")
async def greeting():
    await asyncio.sleep(2)
    return {"message": "Hello World!"}

@app.get("/greeting2")
async def greeting():
    await asyncio.sleep(10)
    return {"message": "Hello New World!"}

@app.post("/submit")
async def submit(data:dict):
    return {"message": "Hello New World!", "data": data['name']}

@app.get("/get_status")
def get_status():
    return {"training":70,"testing":30}

@app.post("/prediction")
async def prediction(payload: dict):
    try:
        X_unknown = [
            payload["sepal-lenght"],
            payload["sepal-width"],
            payload["petal-lenght"],
            payload["petal-width"]
        ]
        X_unknown = numpy.array(X_unknown).reshape(1, -1)
        with open("./model/iris_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        prediction = clf.predict(X_unknown)
        await asyncio.sleep(5)
        return {"predicted_value": prediction[0]}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e.args[0]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training")
async def training():
    trainingModel()
    await asyncio.sleep(15)
    return {"Model":"Model file created"}

