import tensorflow as tf
import pandas as pd
import numpy as np
from google.cloud import storage
import os
from html_model_data_processing import build_and_train_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import re
from bs4 import BeautifulSoup

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
    model = build_and_train_model(dataset_url, model_name)  # your transformer model
    local_path = f"{model_name}.keras"
    model.save(local_path)

    # Upload
    upload_model_to_gcs(local_path, bucket_name, model_name)

# === Load Vocabulary ===
def load_vocab(vocab_path="vocab.txt"):
    with open(vocab_path, "r") as f:
        return f.read().splitlines()

# === Load Vectorizer with Vocab ===
def get_vectorizer(vocab, max_seq_len=160):
    vectorizer = TextVectorization(
        output_mode="int",
        output_sequence_length=max_seq_len,
        standardize=None,
        split="whitespace",
        vocabulary=vocab
    )
    return vectorizer

# === Clean generated HTML ===
def clean_generated_html(text):
    text = re.sub(r'(class="[^"]+")(?=\sclass=")', '', text)
    text = re.sub(r'(<[^ >]+)(?=\s|$)', r'\1>', text)
    text = re.sub(r'>\s+<', '><', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(\s+style="[^"]+"){2,}', lambda m: m.group(1), text)
    try:
        soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.prettify()
    except:
        cleaned = text
    return cleaned.strip()

# === Apply repetition penalty ===
def apply_repetition_penalty(logits, generated_ids, penalty=1.3):
    for idx in set(generated_ids):
        logits[idx] = logits[idx] / penalty
    return logits

# === Generate HTML from prompt ===
def generate_html(prompt, model_path="html_gen_transformer.keras", vocab_path="vocab.txt", max_seq_len=160):
    vocab = load_vocab(vocab_path)
    vectorizer = get_vectorizer(vocab, max_seq_len)
    model = load_model(model_path)

    start_token_idx = vocab.index("[start]")
    end_token_idx = vocab.index("[end]")

    input_tokens = vectorizer(tf.constant([prompt]))
    decoder_input = [start_token_idx]
    generated_ids = [start_token_idx]

    for _ in range(max_seq_len - 1):
        decoder_input_padded = decoder_input + [0] * (max_seq_len - len(decoder_input))
        decoder_tensor = tf.constant([decoder_input_padded])
        preds = model([input_tokens, decoder_tensor], training=False)
        logits = preds[0, len(generated_ids) - 1].numpy()

        logits = apply_repetition_penalty(logits, generated_ids)
        next_token = int(np.argmax(logits))
        if next_token == end_token_idx:
            break
        decoder_input.append(next_token)
        generated_ids.append(next_token)

    tokens = [vocab[idx] for idx in generated_ids if idx not in [0, start_token_idx, end_token_idx]]
    return clean_generated_html(" ".join(tokens))

