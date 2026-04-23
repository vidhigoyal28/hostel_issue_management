# 🛠️ Smart Complaint Management System

An intelligent complaint management system that allows users to register complaints in natural language and automatically processes them using a deep learning model. The system classifies complaints, extracts key details, and prepares structured outputs for efficient resolution.

---

## 🚀 Features

* 📝 **Complaint Registration**

  * Accepts user complaints in plain text

* 🤖 **Automatic Classification**

  * Uses an LSTM-based deep learning model to classify complaints into:

    * Electrical
    * Internet
    * Food
    * Cleanliness
    * Plumbing
    * Maintenance
    * Security

* 📍 **Smart Information Extraction**

  * Extracts:

    * Room Number
    * Floor Number
    * Urgency Level (Low / Medium / High)

* ⚡ **Real-Time API**

  * REST API for seamless integration with mobile/web apps

---

## 🧰 Tech Stack

### 👨‍💻 Core

* **Python**

### 🤖 Machine Learning

* **TensorFlow / Keras** — LSTM model
* **Scikit-learn** — preprocessing & encoding

### 📊 Data Processing

* **Pandas**, **NumPy**

### 🧠 NLP

* **Tokenizer**, **Pad Sequences**

### 🌐 Backend API

* **Flask** — REST API framework

### 🔍 Information Extraction

* **Regex (Rule-based parsing)**

### 📱 Frontend (Planned)

* **Flutter / React**

---

## 🧠 Model Architecture

* Embedding Layer (6000, 64)
* LSTM (64 units, return sequences)
* Dropout (0.3)
* LSTM (32 units)
* Dense (ReLU)
* Output Layer (Softmax)

---

## 📊 Data Pipeline

1. Data loading from CSV
2. Text cleaning & normalization
3. Category standardization
4. Label encoding
5. Tokenization & padding
6. Train-test split (80/20)

---

## 🏋️ Training Details

* Loss: `sparse_categorical_crossentropy`
* Optimizer: `Adam`
* Epochs: 10
* Batch Size: 8

---

## 🌐 API Endpoints

### 🔹 Base URL

```
http://127.0.0.1:5000/
```

---

### 🔹 Health Check

```
GET /
```

**Response:**

```
Complaint Management API is running!
```

---

### 🔹 Predict Complaint

```
POST /predict
```

#### Request Body

```json
{
  "text": "Fan not working in room 203 on 2nd floor, urgent"
}
```

#### Response

```json
{
  "complaint": "Fan not working in room 203 on 2nd floor, urgent",
  "category": "Electrical",
  "room": "203",
  "floor": "2",
  "urgency": "High"
}
```

---

## 🔍 Processing Workflow

1. User sends complaint via app
2. API receives request
3. Text is preprocessed
4. Model predicts category
5. System extracts structured info
6. JSON response is returned

---

## 🖥️ Development Setup

### 1. Install Dependencies

```bash
pip install flask pandas numpy scikit-learn tensorflow
```

### 2. Run the Server

```bash
python app.py
```

### 3. Test API (Postman / Curl)

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d "{\"text\":\"WiFi not working in room 105, urgent\"}"
```

---

## 📁 Project Structure

```
├── app.py              # Flask API + ML model
├── dataset.csv         # Training data
├── README.md
```

---

## ⚠️ Current Limitation

* Model is trained **every time the server starts**, which increases startup time.

---

## 🚀 Future Improvements

* 💾 Save & load trained model (`.h5`)
* ⚡ Faster API response (no retraining)
* 📱 Flutter mobile app integration
* 🗄️ Database (MySQL / Firebase)
* 📊 Admin dashboard
* 🌍 Multilingual support
* 🎯 Priority-based ticket routing

---

## 🎯 Use Cases

* Hostel Management Systems
* College Campuses
* Office Facilities
* Smart Buildings

---

## ⭐ Summary

This project combines **Natural Language Processing (NLP)** and **Deep Learning (LSTM)** with a **Flask API backend** to automate complaint classification and streamline facility management systems.

---
