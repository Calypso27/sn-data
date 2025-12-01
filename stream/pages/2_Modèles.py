import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="2 – Modèles", page_icon="🧠")

st.title("🧠 Modèles de Machine Learning")

st.write(
    """
    Cette page présente les principaux modèles testés pour prédire la qualité des jus :
    - SVM (Support Vector Machine)
    - XGBoost (Gradient Boosting)
    """
)

@st.cache_resource
def load_model_data():
    try:
        model_data = joblib.load("models/juice_model.pkl")
        return model_data
    except Exception:
        return None

model_data = load_model_data()

if model_data is None:
    st.error("Le fichier `models/juice_model.pkl` est introuvable. Entraîne d'abord le modèle.")
else:
    st.success("✅ Modèle final chargé avec succès.")
    st.write("### Modèle sélectionné")
    st.write(f"**Type :** `{type(model_data['model']).__name__}`")
    st.write(f"**Accuracy test :** {model_data['accuracy']:.3f}")

    if "best_params" in model_data:
        st.write("### Hyperparamètres du meilleur modèle")
        st.json(model_data["best_params"])

    st.write("### Variables d'entrée utilisées")
    st.write(model_data.get("feature_names", []))

    # Petite table récap des performances
    st.write("### Récapitulatif des performances (exemple)")
    perf_df = pd.DataFrame(
        [
            {
                "Model": "Meilleur modèle (pickle)",
                "Accuracy": model_data["accuracy"],
            }
        ]
    )
    st.dataframe(perf_df)
