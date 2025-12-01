from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# ===== Chargement du modèle =====
print("📦 Chargement du modèle...")
model_data = joblib.load("juice_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]   # None si modèle = Pipeline SVM
feature_names = model_data.get("feature_names", [])
print("✅ Modèle et scaler chargés avec succès!")

# Ordre des features attendu (même que dans le notebook)
FEATURE_ORDER = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Mapping numérique -> label lisible
LABEL_MAP = {
    0: "Mauvais",
    1: "Moyen",
    2: "Bon",
}

def prepare_features(payload: dict) -> np.ndarray:
    """Construit le vecteur X dans le bon ordre + applique le scaler si nécessaire."""
    values = []
    for feat in FEATURE_ORDER:
        if feat not in payload:
            raise ValueError(f"Champ manquant : {feat}")
        values.append(float(payload[feat]))

    X = np.array(values).reshape(1, -1)

    # Si scaler externe (cas XGBoost), on l'applique
    if scaler is not None:
        X = scaler.transform(X)

    return X


@app.route("/")
def home():
    return """
    <html>
        <head><title>Juice Quality API</title></head>
        <body>
            <h1>🍊 Juice Quality Prediction API</h1>
            <p>Use <code>POST /predict</code> to get predictions</p>
            <p>Check <code>GET /health</code> for API status</p>
        </body>
    </html>
    """


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Préparer les features
        X = prepare_features(data)

        # Prédiction
        y_pred = model.predict(X)[0]

        # Probabilité / confiance si dispo
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            confidence = float(np.max(proba))

        y_int = int(y_pred)
        label = LABEL_MAP.get(y_int, str(y_int))

        return jsonify({
            "success": True,
            "prediction": {
                "label": label,
                "raw": y_int
            },
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    print("🚀 API démarrée sur http://localhost:7860")
    print("📌 Utilisez POST /predict pour faire des prédictions")
    app.run(debug=True, host="0.0.0.0", port=7860)