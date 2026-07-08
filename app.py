"""
app.py
------
Flask web application that serves a form to collect a student's academic
and extracurricular details, feeds them to a trained ML model, and displays
a Placed / Not Placed prediction along with the confidence score.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]

# Human-readable labels + input constraints shown on the form
FIELD_CONFIG = [
    {"name": "cgpa",          "label": "CGPA (out of 10)",          "min": 0,   "max": 10,  "step": 0.01},
    {"name": "tenth_pct",     "label": "10th Percentage (%)",       "min": 0,   "max": 100, "step": 0.1},
    {"name": "twelfth_pct",   "label": "12th Percentage (%)",       "min": 0,   "max": 100, "step": 0.1},
    {"name": "attendance",    "label": "Attendance (%)",            "min": 0,   "max": 100, "step": 0.1},
    {"name": "backlogs",      "label": "Number of Backlogs",        "min": 0,   "max": 20,  "step": 1},
    {"name": "internships",   "label": "Number of Internships",     "min": 0,   "max": 10,  "step": 1},
    {"name": "projects",      "label": "Number of Projects",        "min": 0,   "max": 20,  "step": 1},
    {"name": "communication", "label": "Communication Skill (1-10)","min": 1,   "max": 10,  "step": 0.1},
    {"name": "coding_score",  "label": "Coding/Aptitude Score (0-100)", "min": 0, "max": 100, "step": 0.1},
]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", fields=FIELD_CONFIG, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    errors = []
    values = {}

    # ---- Validate & collect numeric input --------------------------------
    for field in FIELD_CONFIG:
        raw = request.form.get(field["name"], "").strip()
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{field['label']} must be a valid number.")
            continue

        if val < field["min"] or val > field["max"]:
            errors.append(
                f"{field['label']} must be between {field['min']} and {field['max']}."
            )
        values[field["name"]] = val

    if errors:
        return render_template(
            "index.html", fields=FIELD_CONFIG, result=None,
            errors=errors, submitted=request.form
        )

    # ---- Build feature vector in the exact order the model expects -------
    x = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)

    prediction = model.predict(x)[0]
    probability = model.predict_proba(x)[0]
    confidence = round(float(probability[prediction]) * 100, 2)

    result = {
        "label": "Placed ✅" if prediction == 1 else "Not Placed ❌",
        "placed": bool(prediction == 1),
        "confidence": confidence,
    }

    return render_template(
        "index.html", fields=FIELD_CONFIG, result=result, submitted=request.form
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
