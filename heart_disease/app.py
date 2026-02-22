from flask import Flask, request, jsonify
import joblib, pandas as pd
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="assets")
CORS(app)
model = joblib.load("assets/gs_knn.joblib")

feature_cols = ['age','sex','cp','trestbps','chol','fbs',
                'restecg','thalach','exang','oldpeak',
                'slope','ca','thal']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']  # list of values
    # Convert to DataFrame
    df = pd.DataFrame([features], columns=feature_cols)
    
    # Predict
    prob = model.predict_proba(df)[0][1]
    return jsonify({'probability': float(prob)})

if __name__ == "__main__":
    app.run(debug=True)