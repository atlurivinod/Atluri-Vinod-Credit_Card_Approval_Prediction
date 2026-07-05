from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import pickle

app = Flask(__name__)

# Load Trained Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Prediction Form Page
@app.route("/predict_page")
def predict_page():
    return render_template("index.html")

# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["ID"]),
        float(request.form["CODE_GENDER"]),
        float(request.form["FLAG_OWN_CAR"]),
        float(request.form["FLAG_OWN_REALTY"]),
        float(request.form["CNT_CHILDREN"]),
        float(request.form["AMT_INCOME_TOTAL"]),
        float(request.form["NAME_INCOME_TYPE"]),
        float(request.form["NAME_EDUCATION_TYPE"]),
        float(request.form["NAME_FAMILY_STATUS"]),
        float(request.form["NAME_HOUSING_TYPE"]),
        float(request.form["DAYS_BIRTH"]),
        float(request.form["DAYS_EMPLOYED"]),
        float(request.form["FLAG_MOBIL"]),
        float(request.form["FLAG_WORK_PHONE"]),
        float(request.form["FLAG_PHONE"]),
        float(request.form["FLAG_EMAIL"]),
        float(request.form["OCCUPATION_TYPE"]),
        float(request.form["CNT_FAM_MEMBERS"]),
        float(request.form["MONTHS_BALANCE"])
    ]

    prediction = model.predict([features])

    if prediction[0] == 0:
        result = "Credit Card Approved"
    else:
        result = "Credit Card Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)