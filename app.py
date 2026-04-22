from flask import Flask, render_template, request
import pickle
import os
from utils.pdf_extractor import extract_text_from_pdf
from utils.skill_matcher import match_skills

app = Flask(__name__)

model = pickle.load(open("model/saved_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['resume']
    
    
    os.makedirs("uploads", exist_ok=True)

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)

    vec = vectorizer.transform([text])
    role = model.predict(vec)[0]

    score, missing = match_skills(text, role)

    return render_template("result.html",
                           role=role,
                           score=score,
                           missing=missing)

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)