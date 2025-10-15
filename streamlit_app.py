import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests

# Load model
model = joblib.load("model.pkl")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# UI
st.title("🌸 Iris Flower Classifier")

sepal_length = st.slider("Sepal Length (cm)", float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

if st.button("Predict Flower Type"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Flower Type: {iris.target_names[prediction]}")

# Accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.info(f"Model Accuracy on Test Set: {accuracy:.2f}")

# Ollama Query
st.subheader("💬 Ask Ollama about the model or dataset")
user_query = st.text_input("Enter your question:")
if user_query:
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama2",
                "messages": [{"role": "user", "content": user_query}]
            }
        )
        result = response.json()
        st.write("Ollama Response:", result["message"]["content"])
    except Exception as e:
        st.error(f"Failed to connect to Ollama: {e}")


