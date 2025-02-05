from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Diabetes_Prediction.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Form Data:", request.form)  # Debug: Print form data
        features = [
            int(request.form['HighBP']),
            int(request.form['HighChol']),
            int(request.form['CholCheck']),
            float(request.form['BMI']),
            int(request.form['Smoker']),
            int(request.form['Stroke']),
            int(request.form['HeartDiseaseorAttack']),
            int(request.form['PhysActivity']),
            int(request.form['Fruits']),
            int(request.form['Veggies']),
            int(request.form['HvyAlcoholConsump']),
            int(request.form['AnyHealthcare']),
            int(request.form['NoDocbcCost']),
            int(request.form['GenHlth']),
            int(request.form['MentHlth']),
            int(request.form['PhysHlth']),
            int(request.form['DiffWalk']),
            int(request.form['Sex']),
            int(request.form['Age']),
            int(request.form['Education']),
            int(request.form['Income'])
        ]
        print("Extracted Features:", features)  # Debug: Print extracted features

        # Reshape and preprocess features (if required)
        features = np.array(features).reshape(1, -1)
     

        features_scaled = scaler.transform(features)  # Correct: apply existing scaling


        probabilities = model.predict_proba(features_scaled)
        print("Prediction Probabilities:", probabilities)

        # Make prediction
        prediction = model.predict(features_scaled)
        print("Raw Prediction:", prediction)  # Debug: Print raw prediction
        
        # Map prediction to result
        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"

        prob_no_diabetes = probabilities[0][0] * 100  # Probability of class 0 (No Diabetes)
        prob_diabetes = probabilities[0][1] * 100  # Probability of class 1 (Diabetes)

        if prediction[0] == 0:
            message = f"The App predicts a {prob_no_diabetes:.2f}% likelihood of NOT having Diabetes. Keep maintaining a healthy lifestyle!"
        else:
            message = f"The App predicts a {prob_diabetes:.2f}% likelihood of having Diabetes. Please consult a medical professional for further guidance."

        return render_template('result.html', prediction=result, probability_message=message)


        return render_template('result.html', prediction=result)
    except Exception as e:
        print("Error:", e)  # Debug: Print any errors
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)