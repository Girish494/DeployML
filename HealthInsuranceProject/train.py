import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from utils.featureenginnering import create_features
from sklearn.impute import SimpleImputer
import pickle
df=pd.read_csv('C:\MLProject\HealthInsuranceProject\data\insurance_100k_dataset.csv')

df=create_features(df)


X=df.drop('insurance_premium_category',axis=1)
y=df['insurance_premium_category']


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

cat_col=X_train.select_dtypes(include=['object']).columns.tolist()
num_col=X_train.select_dtypes(include=['number']).columns.tolist()

num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='mean')),('scaler',StandardScaler())])
cat_pipeline=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore',drop='first'))])
preprocessor=ColumnTransformer(transformers=[('num',num_pipeline,num_col),('cat',cat_pipeline,cat_col)])

X_train_processed=preprocessor.fit_transform(X_train)
X_test_processed=preprocessor.transform(X_test)

label_encoder = LabelEncoder()
y_train_processed = label_encoder.fit_transform(y_train)
y_test_processed = label_encoder.transform(y_test)
with open('model/label_encoder.pkl','wb')as f:
    pickle.dump(label_encoder,f)
    
X_train_processed=np.asarray(X_train_processed.toarray() if hasattr(X_train_processed,'toarray')else X_train_processed)
X_test_processed=np.asarray(X_test_processed.toarray() if hasattr(X_test_processed,'toarray')else X_test_processed)

X_train_tensor=torch.tensor(X_train_processed,dtype=torch.float32)
X_test_tensor=torch.tensor(X_test_processed,dtype=torch.float32)

y_train_tensor=torch.tensor(y_train_processed,dtype=torch.long)
y_test_tensor=torch.tensor(y_test_processed,dtype=torch.long)

from torch.utils.data import DataLoader,TensorDataset

train_dataset=TensorDataset(X_train_tensor,y_train_tensor)
test_dataset=TensorDataset(X_test_tensor,y_test_tensor)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32)

with open('model/preprocessor.pkl','wb')as f:
    pickle.dump(preprocessor,f)

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


input_dim=X_train_tensor.shape[1]
model=ANNModel(input_dim)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)

model_info={"input_dim":input_dim}
with open('model/model_info.pkl','wb')as f:
    pickle.dump(model_info,f)


train_loss=[]
validation_loss=[]
best_validation_loss=float('inf')
epochs=50
for epoch in range(epochs):
    model.train()
    running_loss=0.0
    for xb,yb in train_loader:
        optimizer.zero_grad()
        outputs=model(xb)
        loss=criterion(outputs,yb)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    epoch_train_loss=running_loss/len(train_loader)
    train_loss.append(epoch_train_loss)

    model.eval()
    running_validation_loss=0.0
    with torch.no_grad():
        for xb,yb in test_loader:
            outputs=model(xb)
            loss=criterion(outputs,yb)
            running_validation_loss+=loss.item()
        epoch_validation_loss=running_validation_loss/len(test_loader)
        validation_loss.append(epoch_validation_loss)

        print(f"{epoch+1}/{epochs}==> training loss {epoch_train_loss} validation loss {epoch_validation_loss}")
        if epoch_validation_loss<best_validation_loss:
            best_validation_loss=epoch_validation_loss
            torch.save(model.state_dict(),'model/best_model.pt')
            print('based model saved')
model.load_state_dict(torch.load('model/best_model.pt'))
model.eval()
with torch.no_grad():
    train_prediction=model(X_train_tensor)
    test_prediction=model(X_test_tensor)

    train_mse_loss=criterion(train_prediction,y_train_tensor)
    test_mse_loss=criterion(test_prediction,y_test_tensor)
print('training loss',train_mse_loss.item())
print('testing loss',test_mse_loss.item())


import matplotlib.pyplot as plt
loss_df=pd.DataFrame({
    "train_loss":train_loss,
    "valid_loss":validation_loss
})
plt.figure(figsize=(12,8))
plt.plot(loss_df['train_loss'],label='train_loss')
plt.plot(loss_df['valid_loss'],label='valid_loss')
plt.xlabel("Epochs")
plt.ylabel("losses")
plt.legend()
plt.show()




from sklearn.metrics import accuracy_score

model.eval()

predictions = []
actuals = []

with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb)

        _, preds = torch.max(outputs, 1)

        predictions.extend(preds.cpu().numpy())
        actuals.extend(yb.cpu().numpy())

acc = accuracy_score(actuals, predictions)

print(f"Accuracy: {acc:.4f}")

from sklearn.metrics import classification_report

print(
    classification_report(
        actuals,
        predictions,
        target_names=label_encoder.classes_
    )
)













print("Completed")