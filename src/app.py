from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo

with open("knn_classifier_k10_manhattan_distance.sav", "rb") as f:
    model = pickle.load(f)

columns = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

class_dict = {
    "0": "Calidad Baja",
    "1": "Calidad Media",
    "2": "Calidad Alta"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # Recoger los 11 valores de las variables
        inputs = []
        for i in range(1, 12):
            value = float(request.form[f"val{i}"])
            inputs.append(value)
        
        X = pd.DataFrame([inputs], columns= columns)
        prediction = model.predict(X)[0]
        pred_class = class_dict[int(prediction)]
    else:
        pred_class = None
    return render_template("index.html", prediction = pred_class)

if __name__ == "__main__":
    app.run(debug = True)