import re
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
import os

# --------- NLTK stopwords ----------
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk as _nltk
    _nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# --------- Text cleaning function ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# --------- Config ----------
MAX_LEN = 120   # same as training
MODEL_PATH = "bilstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# --------- Load model ----------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

# --------- Load tokenizer ----------
print("Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded.")

# --------- Load LabelEncoder ----------
if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"{LABEL_ENCODER_PATH} not found! Please create it during training.")
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
print("LabelEncoder loaded. Classes:", le.classes_)

# --------- Flask app ----------
app = Flask(__name__, template_folder="templates")

# Home route (web UI)
@app.route('/')
def index():
    return render_template('index.html', labels=list(le.classes_))


# API route (JSON)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please provide JSON with key 'text'."}), 400

    text = data["text"]
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(pad)[0]
    idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([idx])[0]

    probs_dict = {le.classes_[i]: float(probs[i]) for i in range(len(probs))}

    return jsonify({
        "text": text,
        "prediction": pred_label,
        "class_index": idx,
        "probs": probs_dict
    })

# Form submission route (web UI)
@app.route("/predict", methods=["POST"])
def web_predict():
    text = request.form.get("text", "")
    if not text:
        return render_template("index.html", labels=list(le.classes_), error="Please enter some text.")

    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(pad)[0]
    idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([idx])[0]

    probs_dict = {le.classes_[i]: float(probs[i]) for i in range(len(probs))}
    return render_template("index.html", labels=list(le.classes_), result=str(pred_label), probs=probs_dict, original=text)

# Run Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)