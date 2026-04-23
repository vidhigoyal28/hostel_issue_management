import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# 1. LOAD BALANCED DATASET
# =============================================================================
DATA_PATH = "dataBalanced.xlsx"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_excel(DATA_PATH)

# The notebook used 'Balanced Dataset' sheet
try:
    df = pd.read_excel(DATA_PATH, sheet_name="Balanced Dataset")
except Exception:
    df = pd.read_excel(DATA_PATH)

df = df[['Complaint_Description', 'Category']].dropna()
df = df.rename(columns={'Complaint_Description': 'text', 'Category': 'category'})

# Basic text cleaning
df['text'] = df['text'].str.lower().str.strip()

# Drop rows with very short or meaningless text
df = df[df['text'].str.len() > 3]

print("Dataset shape:", df.shape)
print("\nCategory distribution:")
print(df['category'].value_counts())
print(f"\nTotal classes: {df['category'].nunique()}")

# =============================================================================
# 2. ENCODE LABELS
# =============================================================================
encoder = LabelEncoder()
y = encoder.fit_transform(df['category'])

print("\nLabel mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"  {i} -> {label}")

# =============================================================================
# 3. TOKENIZATION
# =============================================================================
MAX_WORDS = 8000
MAX_LEN   = 60

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"\nVocab size: {len(tokenizer.word_index)}")
print(f"Input shape: {X_pad.shape}")

# =============================================================================
# 4. TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# =============================================================================
# 5. CLASS WEIGHTS
# =============================================================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# =============================================================================
# 6. MODEL Architecture
# =============================================================================
NUM_CLASSES = len(encoder.classes_)

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=64),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =============================================================================
# 7. TRAIN
# =============================================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# =============================================================================
# 8. EVALUATE
# =============================================================================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[OK] Test Accuracy : {accuracy * 100:.2f}%")
print(f"   Test Loss     : {loss:.4f}")

# =============================================================================
# 9. SAVE MODEL + ARTIFACTS
# =============================================================================
model.save("model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n[OK] Model saved to model.h5")
print("[OK] Tokenizer saved to tokenizer.pkl")
print("[OK] Encoder saved to encoder.pkl")
