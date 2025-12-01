import os
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
        pages_dir = os.path.dirname(__file__)              
        stream_dir = os.path.dirname(pages_dir)            


        model_path = os.path.join(stream_dir, "models", "juice_model.pkl")

        

        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        st.error(f"Modèle non trouvé ou erreur de chargement : {e}")
        return None

model_data = load_model_data()

if model_data is None:
    st.error("Le fichier `stream/models/juice_model.pkl` est introuvable. Entraîne d'abord le modèle.")
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
