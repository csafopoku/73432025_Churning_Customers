from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('model3.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    selected_features = ['gender', 'SeniorCitizen', 'Partner', 'tenure', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                        'PaymentMethod', 'MonthlyCharges']
    
    input_features = [float(request.form[feature]) for feature in selected_features]

    # Preprocess the input data as needed (scaling, encoding, etc.)
    input_data = np.array(input_features).reshape(1, -1)  # Reshape for prediction

    # Make predictions using the loaded model
    prediction_prob = model.predict(input_data)
    
   
    threshold = 0.5  
    prediction = 1 if prediction_prob > threshold else 0

    return render_template('predictions.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
