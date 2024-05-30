import os
import pandas as pd
import numpy as np
import requests
import joblib

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import uvicorn
from pydantic import BaseModel
from io import BytesIO

from src.paths import MODELS_DIR

ride_model = joblib.load(f"{MODELS_DIR}/taxi-demand-artifact.joblib")['model']
price_model = joblib.load(f"{MODELS_DIR}/dynamic-pricing-artifact.joblib")['model']

# Get MLflow URI from my environment variable (defaults to "http://localhost:5000" if not found)
MLFLOW_SERVICE_URI = os.getenv('MLFLOW_TRACKING_URI', "http://localhost:5000")



# Create an instance of FastAPI
app = FastAPI()


@app.get('/')
async def root():
    return "Welcome To Real-Time Spatio-Temporal Demand Forecasting & Price Optimization with Dynamic Pricing"


@app.post('/demand_forecast')
async def batch_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = BytesIO(content)
        df = pd.read_parquet(data)

        inference_request = {"dataframe_records": df.to_dict(orient="records")}
        endpoint = f"{MLFLOW_SERVICE_URI}/invocations"
        response = requests.post(endpoint, json=inference_request)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code,
                                detail=f"Error from MLflow server: {response.text}")

        predictions = response.json().get('predictions', [])
        ride_demand = [np.round(np.expm1(pred)) for pred in predictions]

        return {"predictions": ride_demand}

    except HTTPException as http_exc:
        # Re-raise the HTTPException to be handled by FastAPI
        raise http_exc
        

    except Exception as exc:
        print(f"An unexpected error occurred: {str(exc)}")

        return JSONResponse(status_code=500, content={"message": "An error occurred while processing the request."})



@app.post('/predict_price')
async def batch_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = BytesIO(content)
        df = pd.read_parquet(data)

        price_predictions = np.round(price_model.predict(df), 2).tolist()

        return {"predictions": price_predictions}

    except Exception as exc:
        print(f"An unexpected error occurred: {str(exc)}")
        return JSONResponse(status_code=500, content={"message": "An error occurred while processing the request."})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
