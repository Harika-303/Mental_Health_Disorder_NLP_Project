## ================== IMPORTS ==================
import re
import numpy as np
import pandas as pd
import pickle    # ✅ Required for saving tokenizer and LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords

# ================== DATA PREP ==================
df = pd.read_csv("mental_disorders.csv")

# Keep only relevant labels
df = df[df['status'].isin(['BPD', 'Anxiety', 'depression', 'mentalillness', 'bipolar', 'schizophrenia'])]
df = df.dropna(subset=['title'])

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text) # remove punctuation/numbers
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df["clean_text"] = df["title"].apply(clean_text)

# ================== LABEL ENCODING ==================
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["status"])
num_classes = len(le.classes_)

# Save LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("✅ Saved label_encoder.pkl with classes:", le.classes_)

# Save label classes separately (for Flask app)
np.save('label_classes.npy', le.classes_)
print("✅ Saved label_classes.npy with classes:", le.classes_)

# ================== FEATURES & LABELS ==================
x = df["clean_text"]
y = df["label_encoded"]

# ================== TRAIN-TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)

# ================== TOKENIZER ==================
max_words = 20000
max_len = 120

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved as tokenizer.pkl")

# ================== CLASS WEIGHTS ==================
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

# ================== MODEL ==================
embedding_dim = 128

model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=True)),
    GlobalMaxPool1D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-3),
    metrics=['accuracy']
)
model.build(input_shape=(None,max_len))

model.summary()

# ================== TRAIN ==================
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.1,
    epochs=3,
    batch_size=64,
    class_weight=class_weights,
    verbose=1
)

# ================== EVALUATE ==================
y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n✅ Test Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ================== SAVE MODEL ==================
model.save("bilstm_model.h5")
print("✅ Model saved as bilstm_model.h5")

# ================== SAMPLE PREDICTIONS ==================
test_samples = [
    {"text": "I feel nervous and my heart races whenever I have to speak in public.", "actual": "Anxiety"},
    {"text": "I often get sudden intense emotions and fear being abandoned.", "actual": "BPD"},
    {"text": "I experience periods of extreme energy followed by deep sadness.", "actual": "bipolar"},
    {"text": "I have lost interest in everything and feel hopeless all the time.", "actual": "depression"},
    {"text": "I hear voices that others don’t and sometimes feel people are controlling me.", "actual": "schizophrenia"},
    {"text": "I don’t feel connected to reality and sometimes struggle to identify my feelings.", "actual": "mentalillness"}
]

def predict_disorder(text):
    text_clean = clean_text(text)
    enc = tokenizer.texts_to_sequences([text_clean])
    enc_pad = pad_sequences(enc, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(enc_pad)
    pred_label = np.argmax(pred, axis=1)[0]
    return le.classes_[pred_label]

for sample in test_samples:
    predicted = predict_disorder(sample["text"])
    print(f"Text: {sample['text']}")
    print(f"Actual Disorder: {sample['actual']}")
    print(f"Predicted Disorder: {predicted}")
    print("✅ Correct Prediction!" if predicted == sample["actual"] else "❌ Misclassified")
    print("-" * 100)

