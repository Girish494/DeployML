import numpy as np
import pandas as pd

import re
import joblib

df=pd.read_csv(r"C:\PrimeBatch\MachineLearning\NLP\train.txt",sep=';',header=None,names=['text','emotions'])

df['emotions'].value_counts()
df.isna().sum()

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df['emotions']=label_encoder.fit_transform(df['emotions'])

joblib.dump(label_encoder,"saved_models/label_encoder.pkl")

from app.preprocessing import TextPreprocessor


X=df['text']
y=df['emotions']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

preprocessor=TextPreprocessor()
X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)
joblib.dump(preprocessor,"saved_models/preprocessor.pkl")


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,classification_report,precision_score,recall_score
model=LinearSVC(random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
joblib.dump(model,"saved_models/model.pkl")

print("accuracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred,average='weighted'))
print("recall",recall_score(y_test,y_pred,average='weighted'))
print("\nclassification report\n",classification_report(y_test,y_pred))










print("working")