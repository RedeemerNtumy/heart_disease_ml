from flask import Flask, request, jsonify
import joblib, pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app,origins=["https://heartdiseasepredictorml.netlify.app"])

model = joblib.load("assets/gs_knn.joblib")

feature_cols = ['age','sex','cp','trestbps','chol','fbs',
                'restecg','thalach','exang','oldpeak',
                'slope','ca','thal']

@app.route("/ping")
def ping():
    return "pong"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data['features']], columns=feature_cols)
    prob = model.predict_proba(df)[0][1]
    return jsonify({'probability': float(prob)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
