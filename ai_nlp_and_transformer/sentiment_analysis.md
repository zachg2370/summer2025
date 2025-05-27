Here’s a detailed step-by-step guide for **Project 3: Sentiment Analysis (NLP)** using both traditional NLP techniques and modern transformer-based approaches:

---

### **Step 1: Set Up Environment**
1. **Install Libraries**:
   ```bash
   pip install nltk spacy transformers torch tensorflow datasets pandas matplotlib
   python -m spacy download en_core_web_sm  # SpaCy English model
   python -m nltk.downloader punkt stopwords  # NLTK tokenizers and stopwords
   ```

---

### **Step 2: Load and Explore the Dataset**
Use the **IMDb movie review dataset** (labeled positive/negative):
```python
from datasets import load_dataset

# Load IMDb dataset from Hugging Face
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# Example review and label
print("Review:", train_data[0]["text"])
print("Label:", "Positive" if train_data[0]["label"] else "Negative")
```

---

### **Step 3: Preprocess Text Data**
#### **Tokenization & Cleaning (Traditional NLP)**:
```python
import spacy
from nltk.corpus import stopwords
import re

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    # Remove HTML tags, special characters
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # Tokenize and lemmatize with SpaCy
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    return " ".join(tokens)

# Apply preprocessing
train_data = train_data.map(lambda x: {"cleaned_text": preprocess(x["text"])})
test_data = test_data.map(lambda x: {"cleaned_text": preprocess(x["text"])})
```

---

### **Step 4: Feature Extraction**
#### **Bag-of-Words (Traditional Approach)**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data["cleaned_text"])
X_test = vectorizer.transform(test_data["cleaned_text"])
y_train = train_data["label"]
y_test = test_data["label"]
```

#### **Word Embeddings (Deep Learning)**:
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data["cleaned_text"])
X_train_seq = tokenizer.texts_to_sequences(train_data["cleaned_text"])
X_test_seq = tokenizer.texts_to_sequences(test_data["cleaned_text"])

max_length = 200
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding="post")
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding="post")
```

---

### **Step 5: Build Models**
#### **1. Traditional Model (Logistic Regression)**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_lr.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

#### **2. LSTM with Word Embeddings**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(64, dropout=0.2),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train_padded, y_train, epochs=5, validation_split=0.2)
```

#### **3. Fine-Tune BERT (Transformers)**:
```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Tokenize data
def tokenize(reviews):
    return tokenizer(
        reviews["text"], padding=True, truncation=True, max_length=256
    )

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

# Convert to TensorFlow datasets
tf_train = model.prepare_tf_dataset(train_data, shuffle=True, batch_size=16)
tf_test = model.prepare_tf_dataset(test_data, batch_size=16)

# Compile and fine-tune
model.compile(optimizer="adam", metrics=["accuracy"])
model.fit(tf_train, epochs=3)
```

---

### **Step 6: Evaluate and Interpret**
#### **Confusion Matrix & Metrics**:
```python
from sklearn.metrics import confusion_matrix, classification_report

# For traditional models
print(classification_report(y_test, y_pred))

# For deep learning models
y_pred_proba = model.predict(X_test_padded)
y_pred = (y_pred_proba > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
```

#### **Visualize Attention (Transformers)**:
Use the `BertViz` library to visualize attention heads in BERT:
```python
from bertviz import head_view
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
sentence = "This movie was a fantastic masterpiece!"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
attention = outputs.attentions

# Visualize
head_view(attention, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
```

---

### **Key Concepts Explained**
1. **Tokenization**:
   - Splitting text into words/subwords (e.g., SpaCy, BERT’s WordPiece).
2. **Word Embeddings**:
   - **Word2Vec/GloVe**: Static embeddings capturing semantic meaning.
   - **Contextual Embeddings (BERT)**: Dynamic embeddings based on sentence context.
3. **RNNs/LSTMs**:
   - Process sequences step-by-step, retaining memory of previous inputs.
4. **Transformers & Attention**:
   - **Self-Attention**: Weights words by importance in context.
   - **Pretraining**: BERT/GPT-2 are pretrained on large text corpora.
   - **Fine-Tuning**: Adapt pretrained models to specific tasks (e.g., sentiment analysis).

---

### **Improvement Strategies**
1. **Hyperparameter Tuning**:
   - Adjust LSTM units, dropout rates, or learning schedules.
2. **Advanced Models**:
   - Try `distilbert` for faster training or `roberta` for higher accuracy.
3. **Data Augmentation**:
   - Use back-translation or synonym replacement to expand training data.

---

### **Resources**
- **Hugging Face Tutorials**: [Fine-Tuning BERT](https://huggingface.co/docs/transformers/training)
- **SpaCy NLP Guide**: [Processing Pipelines](https://spacy.io/usage/processing-pipelines)
- **NLTK Book**: [Natural Language Processing with Python](https://www.nltk.org/book/)

By following these steps, you’ll gain hands-on experience with both traditional NLP techniques and cutting-edge transformer models! 📝🤖
