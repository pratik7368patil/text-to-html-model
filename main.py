import os
import tempfile
import pickle
import re
import numpy as np
from lxml import html as lxml_html
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_URL = "https://storage.googleapis.com/text-to-html-models/perfect-html-generation-v2.keras"
TOKENIZER_URL = "https://storage.googleapis.com/text-to-html-models/tokenizer.pkl"
SEQ_LEN = 64  # sequence length used during training


model_fname = os.path.basename(MODEL_URL)
try:
    MODEL_PATH = tf.keras.utils.get_file(
        fname=model_fname,
        origin=MODEL_URL,
        cache_dir=tempfile.gettempdir(),
        cache_subdir='',
        extract=False
    )
except Exception as e:
    raise RuntimeError(f"Failed to download model from URL {MODEL_URL}: {e}")

tokenizer_fname = os.path.basename(TOKENIZER_URL)
try:
    TOKENIZER_PATH = tf.keras.utils.get_file(
        fname=tokenizer_fname,
        origin=TOKENIZER_URL,
        cache_dir=tempfile.gettempdir(),
        cache_subdir='',
        extract=False
    )
except Exception as e:
    raise RuntimeError(f"Failed to download tokenizer from URL {TOKENIZER_URL}: {e}")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

if not os.path.exists(TOKENIZER_PATH):
    raise RuntimeError(f"Tokenizer file not found: {TOKENIZER_PATH}")
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer from {TOKENIZER_PATH}: {e}")

# Special tokens
START_TOKEN = "[start]"
END_TOKEN   = "[end]"

def generate_html(columns: int, rows: int) -> str:
    indent = '    '
    out = '<div class="layout-container">\n'
    for r in range(rows):
        out += f'{indent}<div class="row" style="display: flex; gap: 10px;">\n'
        for c in range(columns):
            idx = r * columns + c + 1
            out += f'{indent*2}<div class="cell">Cell {idx}</div>\n'
        out += f'{indent}</div>\n'
    out += '</div>'
    return out


def parse_layout_prompt(prompt: str):
    match = re.search(r"(\d+)\s*columns?[\s,]*(\d+)\s*rows?", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def greedy_decode(prompt: str, penalty: float = 1.2) -> str:
    layout = parse_layout_prompt(prompt)
    if layout:
        return generate_html(*layout)
    start_id = tokenizer.word_index.get(START_TOKEN)
    end_id   = tokenizer.word_index.get(END_TOKEN)
    enc_seq = tokenizer.texts_to_sequences([prompt])
    enc_seq = pad_sequences(enc_seq, maxlen=SEQ_LEN, padding='post')
    dec_seq = [start_id]
    for _ in range(SEQ_LEN - 1):
        dec_input = pad_sequences([dec_seq], maxlen=SEQ_LEN, padding='post')
        preds = model.predict([enc_seq, dec_input], verbose=0)
        logits = preds[0, len(dec_seq)-1]
        for idx in set(dec_seq):
            logits[idx] /= penalty
        next_id = int(np.argmax(logits))
        if next_id == end_id:
            break
        dec_seq.append(next_id)
    tokens = [tokenizer.index_word.get(i, '') for i in dec_seq if i not in [0, start_id, end_id]]
    raw_html = ' '.join(tokens)
    try:
        doc = lxml_html.fromstring(raw_html)
        return lxml_html.tostring(doc, pretty_print=True, encoding='unicode').strip()
    except Exception:
        return raw_html

app = FastAPI(title="Layout generation", version="1.0")

class GenerateRequest(BaseModel):
    prompt: str
    penalty: float = 1.2

class GenerateResponse(BaseModel):
    html: str

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def ready_check():
    return {"status": "APIs are ready!"}

@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    html_out = greedy_decode(req.prompt, req.penalty)
    return GenerateResponse(html=html_out)