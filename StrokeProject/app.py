from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import requests
import json
import os

# Flask app
app = Flask(__name__, template_folder="templates")

# Load ML model and label encoders
try:
    with open("stroke_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    print("Error: Model and encoder files not found. Please ensure 'stroke_model.pkl' and 'label_encoders.pkl' are in the same directory.")
    model = None
    encoders = None

# Gemini API config
GEMINI_API_KEY = "AIzaSyC5QBVqPD8dHZ8690SJf6IjJQtILQBPdck"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

def get_gemini_suggestion(prompt_text):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {
            "parts": [
                {
                    "text": "You are a health assistant providing general, non-medical advice. Offer actionable, positive, and simple suggestions for a healthier lifestyle based on the user's information. Do not diagnose or recommend specific treatments. Always advise consulting a healthcare professional for personalized advice."
                }
            ]
        },
    }
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        generated_text = result.get('candidates', [])[0].get('content', {}).get('parts', [])[0].get('text', 'No suggestion generated.')
        return generated_text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Error getting suggestion"

# Routes
@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/verification.html", methods=["GET", "POST"])
def verification():
    if request.method == "POST":
        return redirect(url_for("main"))
    return render_template("verification.html")

@app.route("/main.html", methods=["GET", "POST"])
def main():
    prediction = None
    suggestion = None
    probability = None
    
    if request.method == "POST":
        if model is None or encoders is None:
            prediction = "Error: ML model is not loaded."
            suggestion = "Please check the server logs for missing files."
            return render_template("main.html", prediction=prediction, suggestion=suggestion, probability=probability)
        
        try:
            gender = int(request.form["gender"])
            age = float(request.form["age"])
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            ever_married = int(request.form["ever_married"])
            work_type = int(request.form["work_type"])
            residence = request.form["residence"]
            glucose = float(request.form["glucose"])
            bmi = float(request.form["bmi"])
            smoking = int(request.form["smoking"])
            
            residence_encoded = encoders["Residence_type"].transform([residence])[0]
            X_input = np.array([[gender, age, hypertension, heart_disease, ever_married,
                                 work_type, residence_encoded, glucose, bmi, smoking]])
            
            pred = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1]
            
            if pred == 1:
                prediction_text = "High risk of Stroke!"
                prompt = f"The patient has high stroke risk. Age: {age}, Hypertension: {hypertension}, Heart: {heart_disease}, Glucose: {glucose}, BMI: {bmi}, Smoking: {smoking}."
            else:
                prediction_text = "Low risk of Stroke"
                prompt = f"The patient has low stroke risk. Age: {age}, Hypertension: {hypertension}, Heart: {heart_disease}, Glucose: {glucose}, BMI: {bmi}, Smoking: {smoking}."
                
            suggestion = get_gemini_suggestion(prompt)
            prediction = prediction_text
            
        except Exception as e:
            prediction = f"Error: {e}"
            suggestion = "Check your input values."
    
    return render_template("main.html", prediction=prediction, suggestion=suggestion, probability=probability)

# Vercel-compatible run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
