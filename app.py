from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load('mymodel.joblib')

# Initialize LabelEncoder and fit it on known countries
label_encoder = LabelEncoder()
known_countries = ['Afghanistan', 'India', 'USA']  # Add more known countries as needed
label_encoder.fit(known_countries)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling form submissions and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        country = request.form.get('country', '').strip()
        year = request.form.get('year')
        schi = request.form.get('schi')
        bipo_dis = request.form.get('bipo_dis')
        eat_dis = request.form.get('eat_dis')
        anx = request.form.get('anx')
        drug_use = request.form.get('drug_use')
        depr = request.form.get('depr')
        alch = request.form.get('alch')
        
        # Validate and parse inputs
        try:
            year = int(year)
            schi = float(schi)
            bipo_dis = float(bipo_dis)
            eat_dis = float(eat_dis)
            anx = float(anx)
            drug_use = float(drug_use)
            depr = float(depr)
            alch = float(alch)
        except ValueError:
            return "Invalid input data. Please provide valid values.", 400
        
        # Transform country to numeric using the pre-fitted encoder
        try:
            country_encoded = label_encoder.transform([country])[0]
        except ValueError:
            return f"Country '{country}' is not recognized. Please provide a valid country.", 400

        # Make a prediction
        prediction = model.predict([[country_encoded, year, schi, bipo_dis, eat_dis, anx, drug_use, depr, alch]])

        # Format the prediction as a percentage string with 2 decimal places
        prediction_str = f"{prediction[0]:.2f}%"

        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction_str)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request. Please check your inputs and try again.", 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
