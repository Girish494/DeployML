from pathlib import Path
import joblib
import torch
import torch.nn as nn
import pickle
BASE_DIR=Path(__file__).resolve().parent
PROJECT_ROOT=BASE_DIR.parent
MODEL_DIR=PROJECT_ROOT/"model"
MODEL_PATH=MODEL_DIR/"best_model.pt"
PREPROCESSOR_PATH=MODEL_DIR/"preprocessor.pkl"
MODEL_INFO_PATH=MODEL_DIR/"model_info.pkl"

preprocessor=joblib.load(PREPROCESSOR_PATH)
model_info=joblib.load(MODEL_INFO_PATH)
input_dim=model_info['input_dim']
with open(MODEL_DIR/'label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
class ANNModel(nn.Module):
    def __init__(self,input_dim):
        super(ANNModel,self).__init__()
        self.model = nn.Sequential(
        nn.Linear(input_dim,64),
        nn.ReLU(),
       
        nn.Linear(64,32),
        nn.ReLU(),
       
        nn.Linear(32,16),
        nn.ReLU(),

        nn.Linear(16,3)
    )
    def forward(self,x):
        return self.model(x)

model=ANNModel(input_dim)
model.load_state_dict(torch.load(MODEL_PATH,map_location='cpu'))
model.eval()