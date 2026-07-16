from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

@app.route("/")
def home():
    return "House Price Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    city_encoded = encoder.transform(df[["city"]])

    df = df.drop("city", axis=1)

    city_df = pd.DataFrame(
        city_encoded,
        columns=encoder.get_feature_names_out(["city"])
    )

    final_df = pd.concat(
        [df.reset_index(drop=True), city_df.reset_index(drop=True)],
        axis=1
    )

    prediction = model.predict(final_df)

    return jsonify({"Predicted Price": float(prediction[0])})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
