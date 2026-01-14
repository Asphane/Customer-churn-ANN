# 🚀 Customer Churn Prediction using Artificial Neural Network (ANN)

An **end-to-end Machine Learning project** that predicts whether a customer is likely to leave a bank using an **Artificial Neural Network (ANN)**.  
This project covers **data preprocessing, model building, evaluation, and deployment** — all wrapped into a clean and interactive **Streamlit web application**.

---

## 🎯 Project Objective

Customer churn directly impacts business revenue.  
The goal of this project is to **identify customers who are likely to churn** so that proactive retention strategies can be applied.

This system takes customer attributes as input and outputs a **churn probability** using a trained ANN model.

---

## 🧠 What This Project Covers

### 🔹 Data Preprocessing
- Loaded the `Churn_Modelling.csv` dataset
- Removed non-predictive columns:
  - `RowNumber`
  - `CustomerId`
  - `Surname`
- Encoded categorical features:
  - `Gender` → Label Encoding
  - `Geography` → One-Hot Encoding (France, Germany, Spain)
- Applied **Standard Scaling** to numerical features
- Saved all preprocessing objects for production use:
  - `scaler.pkl`
  - `label_encoder_gender.pkl`
  - `one_hot_encoder_geography.pkl`

---

### 🔹 Model Architecture (Artificial Neural Network)

A **Sequential ANN** built using TensorFlow / Keras:

- **Input Layer:** 12 neurons (matching encoded feature count)
- **Hidden Layer 1:** 64 neurons, ReLU activation
- **Hidden Layer 2:** 32 neurons, ReLU activation
- **Output Layer:** 1 neuron, Sigmoid activation (churn probability)

---

### 🔹 Training & Optimization

- **Optimizer:** Adam (learning rate = 0.01)
- **Loss Function:** Binary Crossentropy
- **Callbacks Used:**
  - Early Stopping (monitored `val_loss`)
  - TensorBoard for training visualization
- Prevented overfitting by restoring the **best model weights**

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| Training Accuracy | ~88.27% |
| Validation Accuracy | ~85% – 86% |
| Training Loss | ~0.279 |
| Validation Loss | ~0.403 |

### 🔍 Performance Insight
- Strong generalization performance
- Very small gap (≈2–3%) between training and validation accuracy
- Early Stopping effectively controlled overfitting

---

## 🌐 Deployment (Streamlit Web App)

The trained ANN model is deployed using **Streamlit**, allowing:

- Real-time churn prediction
- User-friendly input interface
- Consistent preprocessing using saved encoders & scaler
- Instant probability-based prediction output

📁 Deployment file:
```
app.py
```

---

## 📁 Project Structure

```

Customer-Churn-Prediction/
│
├── main.ipynb                         # Model training & evaluation
├── app.py                             # Streamlit deployment app
├── Churn_Modelling.csv                # Dataset
│
├── model.h5                           # Trained ANN model
├── scaler.pkl                         # StandardScaler
├── label_encoder_gender.pkl           # Gender encoder
├── one_hot_encoder_geography.pkl      # Geography encoder
│
└── logs/                              # TensorBoard logs

```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install tensorflow scikit-learn pandas numpy streamlit matplotlib
````

### 2️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🏆 Key Highlights

* End-to-end ML pipeline
* Production-ready preprocessing
* ANN-based binary classification
* Overfitting control with Early Stopping
* Real-time deployment using Streamlit
* Clean and modular project structure

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Class imbalance handling (SMOTE)
* Feature importance analysis
* Model comparison with XGBoost / Random Forest
* Cloud deployment (AWS / GCP)

---

## ⭐ Final Note

This project demonstrates a **complete real-world ML workflow** — from raw data to deployment.
If you find this useful, consider **starring ⭐ the repository**!

Happy Learning & Predicting! 🧠✨

```
```
