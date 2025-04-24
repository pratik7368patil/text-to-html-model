import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from bs4 import BeautifulSoup
import re
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


df = []
max_seq_len = 10
vocab = []
vocab_size = len(vocab)
vectorizer = None
prompts = []
outputs = []
all_text = []
dataset = []
model = None

def extract_label_from_prompt(prompt):
    # Use all capitalized or non-stopword tokens as label candidates
    words = prompt.strip().split()
    for word in reversed(words):
        w = word.strip(".,!?:;'").capitalize()
        if len(w) > 1 and w.isalpha():
            return w
    return "Submit"

def inject_text_into_html(row):
    soup = BeautifulSoup(row["output"], "html.parser")
    label = extract_label_from_prompt(row["prompt"])
    for tag in soup.find_all("button"):
        tag.string = label
    return str(soup)

def read_data(dataset_url):
    return pd.read_csv(dataset_url)

def data_cleanup():
  df = df.dropna(subset=["prompt", "output"])
  df["output"] = df.apply(inject_text_into_html, axis=1)

  # Add special tokens
  start_token, end_token = "[start]", "[end]"
  df["output"] = df["output"].apply(lambda x: f"{start_token} {x} {end_token}")

  prompts = df["prompt"].astype(str).tolist()
  outputs = df["output"].astype(str).tolist()
  all_text = prompts + outputs

def input_extraction_and_tokenization():
  max_seq_len = max(len(txt.split()) for txt in prompts + outputs)

  print("max_seq_len: ", max_seq_len)
  max_seq_len = 160 if max_seq_len > 160 else max_seq_len
  print("updated max_seq_len: ", max_seq_len)

  vectorizer = TextVectorization(
      output_mode='int',
      output_sequence_length=max_seq_len,
      standardize=None,
      split='whitespace'
  )
  vectorizer.adapt(all_text)

  vocab = vectorizer.get_vocabulary()
  vocab_size = len(vocab)
  print('Vocab length: ', vocab_size)

def format_dataset(prompt, output):
    enc_tokens = vectorizer(prompt)
    dec_tokens = vectorizer(output)
    dec_input = tf.concat([[0], dec_tokens[:-1]], axis=0)  # shift right
    return (enc_tokens, dec_input), dec_tokens

def create_dataset():
    dataset = tf.data.Dataset.from_tensor_slices((prompts, outputs))
    dataset = dataset.map(lambda p, o: format_dataset(p, o))
    dataset = dataset.shuffle(64).batch(16).prefetch(tf.data.AUTOTUNE)

    print('Dataset is ready.')

def transformer_model(vocab_size, seq_len):
    enc_inputs = layers.Input(shape=(seq_len,), dtype="int64")
    dec_inputs = layers.Input(shape=(seq_len,), dtype="int64")

    embed = layers.Embedding(vocab_size, 256)
    enc_emb = embed(enc_inputs)
    dec_emb = embed(dec_inputs)

    pos_enc = layers.Embedding(seq_len, 256)
    enc_emb += pos_enc(tf.range(start=0, limit=seq_len))
    dec_emb += pos_enc(tf.range(start=0, limit=seq_len))

    for _ in range(2):
        attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=256)(dec_emb, enc_emb)
        x = layers.LayerNormalization()(attn_out + dec_emb)
        ffn = layers.Dense(512, activation="relu")(x)
        ffn = layers.Dense(256)(ffn)
        dec_emb = layers.LayerNormalization()(ffn + x)

    outputs = layers.Dense(vocab_size, activation="softmax")(dec_emb)
    return tf.keras.Model([enc_inputs, dec_inputs], outputs)

def create_and_compile_model():
    model = transformer_model(vocab_size, max_seq_len)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()


def train_with_dataset():
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("best_model.keras", save_best_only=True)
    ]

    history = model.fit(dataset, epochs=100, callbacks=callbacks)
    return history

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

def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    for idx in set(generated_ids):
        logits[idx] = logits[idx] / penalty
    return logits

# === Greedy Decoding with Repetition Penalty ===
def generate_html(prompt, seq_length=max_seq_len):
    input_tokens = vectorizer(tf.constant([prompt]))
    start_token_idx = vocab.index("[start]")
    end_token_idx = vocab.index("[end]")

    decoder_input = [start_token_idx]
    generated_ids = [start_token_idx]

    for _ in range(seq_length - 1):
        decoder_input_padded = decoder_input + [0] * (seq_length - len(decoder_input))
        decoder_tensor = tf.constant([decoder_input_padded])
        preds = model([input_tokens, decoder_tensor], training=False)
        logits = preds[0, len(generated_ids) - 1].numpy()

        # Apply repetition penalty
        logits = apply_repetition_penalty(logits, generated_ids, penalty=1.3)

        next_token = int(np.argmax(logits))
        if next_token == end_token_idx:
            break

        decoder_input.append(next_token)
        generated_ids.append(next_token)

    tokens = [vocab[idx] for idx in generated_ids if idx not in [0, start_token_idx, end_token_idx]]
    return clean_generated_html(" ".join(tokens))

def save_model(model_name):
    model.save(f"{model_name}.keras")


def build_and_train_model(data_url, model_name):
    df = read_data(data_url)
    data_cleanup()
    input_extraction_and_tokenization()
    format_dataset()
    create_dataset()
    transformer_model()
    create_and_compile_model()
    train_with_dataset()
    save_model(model_name)
    return model