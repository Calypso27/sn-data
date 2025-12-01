import streamlit as st
import joblib
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Qualité de Jus – Application",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2405/2405477.png", width=100)
    st.title("🍊 Juice Quality")
    st.markdown("---")
    st.markdown(
        """
        **Analyse et prédiction de la qualité de jus**

        Cette application permet de :
        - Explorer les données
        - Entraîner et comparer des modèles
        - Faire des prédictions locales
        - Tester une API de prédiction déployée
        """
    )

@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("models/juice_model.pkl")
        return model_data
    except Exception:
        return None

model_data = load_model()

st.markdown("<h1 class='main-title'>🍊 Application de Prédiction de Qualité de Jus</h1>", unsafe_allow_html=True)

if model_data:
    st.success("✅ Modèle final chargé avec succès.")
    st.metric("Accuracy du modèle (test)", f"{model_data['accuracy']:.1%}")
else:
    st.warning("⚠️ Modèle non trouvé. Va d'abord sur la page modèles/entraînement pour le générer.")

st.write(
    """
    Navigue dans le menu latéral pour accéder aux différentes sections :
    1. **Exploration** : aperçu du jeu de données et statistiques descriptives.
    2. **Modèles** : description du modèle final et de ses hyperparamètres.
    3. **Prédiction locale** : saisie de caractéristiques et prédiction via le modèle en mémoire.
    4. **Test API** : appel de l'API déployée sur Hugging Face.
    """
)
