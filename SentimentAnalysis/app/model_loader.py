from app.preprocessing import TextPreprocessor
import joblib
model=joblib.load(r"C:\PrimeBatch\MachineLearning\NLP\saved_models\model.pkl")
preprocessor=joblib.load(r"C:\PrimeBatch\MachineLearning\NLP\saved_models\preprocessor.pkl")
label_encoder=joblib.load(r"C:\PrimeBatch\MachineLearning\NLP\saved_models\label_encoder.pkl")

print("working")