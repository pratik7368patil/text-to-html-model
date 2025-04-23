from fastapi import FastAPI
import tensorflow as tf
import pandas as pd
import numpy as np
from pydantic import BaseModel
from google.cloud import storage

app = FastAPI()

class Status(BaseModel):
    tensorflow_version: str
    pandas_version: str
    numpy_version: str
    packages_loaded: bool

@app.get("/")
async def health_check():
    return {
        "status": "running",
        "package_versions": {
            "tensorflow": tf.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        }
    }

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from model_utils import train_model_and_upload, generate_html
import os

app = FastAPI()

class TrainRequest(BaseModel):
    dataset_url: str   # or use a default dataset
    model_name: str

class GenerateRequest(BaseModel):
    prompt: str
    model_name: str

@app.post("/train")
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model_and_upload, request.dataset_url, request.model_name)
    return {"message": f"Training started for {request.model_name}"}

@app.post("/generate")
def generate(req: GenerateRequest):
    html = generate_html(req.prompt, req.model_name)
    return {"html": html}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
