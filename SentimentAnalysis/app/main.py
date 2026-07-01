from fastapi import FastAPI
from app.schema import TextRequest
from app.model_loader import (model,preprocessor,label_encoder)
app=FastAPI()

@app.get("/")
def home():
    return {"message":"Emotion Detection API"}

@app.post("/")
def predict(data:TextRequest):
    X=preprocessor.transform([data.text])
    prediction=model.predict(X)
    emotion=label_encoder.inverse_transform(prediction)
    return{
        "text":data.text,
        "prediction":emotion[0]
    }



print("working")