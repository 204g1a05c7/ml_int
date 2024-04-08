from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib

# Load the trained model
model = joblib.load("your_model2.keras")

# Define your FastAPI app
app = FastAPI()

# Define the request body schema
class Item(BaseModel):
    external_status: str

# Define a dictionary mapping external statuses to internal statuses
status_mapping = {
    "port out": "port out",
    "terminal in": "inbound terminal",
    "port in" : "port in",
    "gate in' : "gate in",
    "on rail" :"on rail",
    # Add more mappings as needed
}

# Initialize label encoders
ext_enc = LabelEncoder()

# Define the endpoint for making predictions
@app.post("/predict/")
async def predict(item: Item):
    # Get the internal status corresponding to the external status
    internal_status = status_mapping.get(item.external_status, "Unknown")
    
    # Preprocess the input data (encode the external status)
    external_status_encoded = ext_enc.transform([internal_status])[0]

    # Make prediction using the internal status
    prediction = model.predict([external_status_encoded])
    
    return {"external_status": item.external_status, "internal_status": internal_status, "predicted_internal_status": prediction[0]}
