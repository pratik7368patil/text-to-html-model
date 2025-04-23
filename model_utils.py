import tensorflow as tf
import pandas as pd
import numpy as np
from google.cloud import storage
import os

def upload_model_to_gcs(model_path, bucket_name, model_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{model_name}.keras")
    blob.upload_from_filename(model_path)

def download_model_from_gcs(model_name, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{model_name}.keras")
    local_path = f"./models/{model_name}.keras"
    blob.download_to_filename(local_path)
    return tf.keras.models.load_model(local_path)

def train_model_and_upload(dataset_url, model_name, bucket_name="html-models"):
    # Load dataset
    df = pd.read_csv(dataset_url)
    # Your preprocessing, vectorization, training logic here...
    model = build_model(...)  # your transformer model
    model.fit(...)  # training
    local_path = f"{model_name}.keras"
    model.save(local_path)

    # Upload
    upload_model_to_gcs(local_path, bucket_name, model_name)

def generate_html(prompt, model_name, bucket_name="html-models"):
    # Load model if not cached
    local_path = f"./models/{model_name}.keras"
    if not os.path.exists(local_path):
        download_model_from_gcs(model_name, bucket_name)
    model = tf.keras.models.load_model(local_path)

    # Run your tokenization + inference logic
    return "<button>Submit</button>"  # placeholder
