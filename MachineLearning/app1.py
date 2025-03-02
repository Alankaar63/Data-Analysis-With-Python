from flask import Flask, jsonify, request
import joblib
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "ML Model is running!"

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features'].reshape(-1,1))
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)``
    
