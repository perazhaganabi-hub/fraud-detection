from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# 🔥 Safe model loading (important for Render)
try:
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = pickle.load(open(model_path, "rb"))
except Exception as e:
    print("Model loading error:", e)
    model = None

# 🔹 Home page
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return "❌ Model not loaded properly"

        amount = float(request.form.get("amount", 0))
        location = int(request.form.get("location", 0))
        time = int(request.form.get("time", 0))
        device = int(request.form.get("device", 0))
        transactions = int(request.form.get("transactions", 0))

        features = np.array([[amount, location, time, device, transactions]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "⚠️ Fraud Transaction"
        else:
            result = "✅ Legitimate Transaction"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"❌ Error: {str(e)}"

# 🔥 IMPORTANT FOR RENDER DEPLOYMENT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)