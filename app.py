from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained Random Forest model
model = joblib.load("rf_model.pkl")

# Define the request body schema
class PredictionInput(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    b: float
    lstat: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Ensure that all inputs are valid numbers
    if any(value is None or value == "" for value in input_data.dict().values()):
        raise ValueError("One or more inputs are invalid.")
    
    # Log the input data to check if all values are correct
    print(f"Received input data: {input_data}")
    
    # Prepare the input data for prediction
    data = np.array([[input_data.crim, input_data.zn, input_data.indus, input_data.chas, input_data.nox,
                      input_data.rm, input_data.age, input_data.dis, input_data.rad, input_data.tax,
                      input_data.ptratio, input_data.b, input_data.lstat]])

    # Log the input data after conversion to numpy array
    print(f"Input data converted to numpy array: {data}")
    
    # Get the predicted price (in thousands of dollars)
    prediction = model.predict(data)

    # Log the prediction result
    print(f"Model Prediction: {prediction}")

    # Ensure the prediction is a valid number
    predicted_price_in_dollars = prediction[0] * 1000
    if np.isnan(predicted_price_in_dollars):
        print("Prediction returned NaN.")
    
    # Format the price with thousands separator and no decimals
    formatted_price = f"${predicted_price_in_dollars:,.0f}"  # Format with commas and no decimals
    
    return {"predicted_price": formatted_price}  # Return formatted price as string
