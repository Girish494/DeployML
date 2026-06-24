from fastapi import FastAPI
import pandas as pd
import torch
from api.schema import CustomerRequest
from api.model import model,preprocessor,label_encoder
from utils.featureenginnering import create_features

app=FastAPI()
@app.get("/")
def home():
    return {'message':'welcome to insurance prediction system'}
@app.post('/predict')
def predict(data:CustomerRequest):
    df=pd.DataFrame([data.model_dump()])
    df=create_features(df)
    x=preprocessor.transform(df)
    if hasattr(x,'toarray'):
        x=x.toarray()
    x=torch.tensor(x,dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        prediction=model(x)
        pred_class=torch.argmax(prediction,dim=1).item()
    prediction=label_encoder.inverse_transform([pred_class])[0]
    
    return {'prediction':prediction}
print('completed')