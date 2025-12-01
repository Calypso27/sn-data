import os
import streamlit as st
import joblib
import numpy as np
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="4 – Prédiction locale", page_icon="🎯")

st.title("🎯 Prédiction Locale de Qualité de Jus")

@st.cache_resource
def load_model_data():
    # Dossier courant du fichier : SN/stream/pages
    pages_dir = os.path.dirname(__file__)
    # Dossier stream : parent de pages
    stream_dir = os.path.dirname(pages_dir)

    # Modèle situé dans SN/stream/models/juice_model.pkl
    model_path = os.path.join(stream_dir, "models", "juice_model.pkl")

    model_data = joblib.load(model_path)
    return model_data

try:
    model_data = load_model_data()
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    st.stop()

model = model_data["model"]
feature_names = model_data["feature_names"]

st.write("Renseigne les caractéristiques du jus pour obtenir une prédiction à partir du modèle local.")

defaults = {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}

col1, col2, col3 = st.columns(3)
with col1:
    fixed_acidity = st.number_input("Acidité fixe (g/L)", value=defaults["fixed_acidity"])
    volatile_acidity = st.number_input("Acidité volatile (g/L)", value=defaults["volatile_acidity"])
    citric_acid = st.number_input("Acide citrique (g/L)", value=defaults["citric_acid"])
    residual_sugar = st.number_input("Sucre résiduel (g/L)", value=defaults["residual_sugar"])

with col2:
    chlorides = st.number_input("Chlorures (g/L)", value=defaults["chlorides"])
    free_sulfur_dioxide = st.number_input("SO2 libre (mg/L)", value=defaults["free_sulfur_dioxide"])
    total_sulfur_dioxide = st.number_input("SO2 total (mg/L)", value=defaults["total_sulfur_dioxide"])
    density = st.number_input("Densité (g/cm³)", value=defaults["density"], format="%.4f")

with col3:
    pH = st.number_input("pH", value=defaults["pH"])
    sulphates = st.number_input("Sulfates (g/L)", value=defaults["sulphates"])
    alcohol = st.number_input("Alcool (% vol)", value=defaults["alcohol"])

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔮 Prédire (modèle local)"):
    X = np.array(
        [
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol,
        ]
    ).reshape(1, -1)

    y_pred = model.predict(X)[0]
    label_map = {0: "Mauvais", 1: "Moyen", 2: "Bon"}
    label = label_map.get(int(y_pred), str(y_pred))

    st.success(f"Qualité prédite : **{label}** (classe {int(y_pred)})")

    st.session_state.history.append(
        {
            "timestamp": datetime.now(),
            "prediction": label,
        }
    )

if st.session_state.history:
    st.markdown("---")
    st.subheader("Historique des prédictions locales")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df.sort_values("timestamp", ascending=False), use_container_width=True)
