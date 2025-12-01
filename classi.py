import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Prédiction Qualité de Jus",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# STYLES CSS 
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #F7931E 0%, #FF6B35 100%);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# TITRE ET DESCRIPTION
st.markdown('<h1 class="main-header">🍊 Prédiction de Qualité de Jus</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Système de classification basé sur modèle optimisé (SVM/XGBoost)</p>', unsafe_allow_html=True)

# INITIALISATION DE LA SESSION
if "history" not in st.session_state:
    st.session_state.history = []

if "api_url" not in st.session_state:
    st.session_state.api_url = "https://calypso-mb-api-j.hf.space"

# SIDEBAR - CONFIGURATION
with st.sidebar:
    st.header("⚙️ Configuration")

    # URL de l'API
    api_url = st.text_input(
        "URL de l'API",
        value=st.session_state.api_url,
        help="Adresse du serveur Flask"
    )
    st.session_state.api_url = api_url

    # Test de connexion
    if st.button("🔍 Tester la connexion"):
        try:
            response = requests.get(f"{api_url}/health", timeout=3)
            if response.status_code == 200:
                st.success("✅ API connectée !")
            else:
                st.error("❌ API inaccessible")
        except Exception:
            st.error("❌ Impossible de se connecter à l'API")


    st.divider()

    # Historique
    st.header("📈 Historique")
    if st.session_state.history:
        st.write(f"**{len(st.session_state.history)}** prédictions effectuées")
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Aucune prédiction encore")


# FORMULAIRE DE SAISIE
st.header("📋 Saisir les Caractéristiques du Jus")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔬 Acidité")
    fixed_acidity = st.number_input(
        "Acidité Fixe (g/L)",
        min_value=0.0,
        max_value=20.0,
        value=7.4,
        step=0.1,
        help="Acides non volatils présents dans le jus",
    )

    volatile_acidity = st.number_input(
        "Acidité Volatile (g/L)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.01,
        help="Acides volatils (acide acétique)",
    )

    citric_acid = st.number_input(
        "Acide Citrique (g/L)",
        min_value=0.0,
        max_value=1.5,
        value=0.0,
        step=0.01,
        help="Acide citrique naturel",
    )

    pH = st.number_input(
        "pH",
        min_value=2.5,
        max_value=4.5,
        value=3.51,
        step=0.01,
        help="Mesure d'acidité (3-4 optimal)",
    )

with col2:
    st.subheader("🍬 Composition")
    residual_sugar = st.number_input(
        "Sucre Résiduel (g/L)",
        min_value=0.0,
        max_value=20.0,
        value=1.9,
        step=0.1,
        help="Sucres restants après fermentation",
    )

    chlorides = st.number_input(
        "Chlorures (g/L)",
        min_value=0.0,
        max_value=0.2,
        value=0.076,
        step=0.001,
        format="%.3f",
        help="Quantité de sel",
    )

    density = st.number_input(
        "Densité (g/cm³)",
        min_value=0.98,
        max_value=1.01,
        value=0.9978,
        step=0.0001,
        format="%.4f",
        help="Densité du jus",
    )

    sulphates = st.number_input(
        "Sulfates (g/L)",
        min_value=0.0,
        max_value=2.0,
        value=0.56,
        step=0.01,
        help="Additif antioxydant",
    )

with col3:
    st.subheader("💨 Gaz & Alcool")
    free_sulfur_dioxide = st.number_input(
        "SO2 Libre (mg/L)",
        min_value=0,
        max_value=100,
        value=11,
        step=1,
        help="Dioxyde de soufre libre",
    )

    total_sulfur_dioxide = st.number_input(
        "SO2 Total (mg/L)",
        min_value=0,
        max_value=300,
        value=34,
        step=1,
        help="Dioxyde de soufre total",
    )

    alcohol = st.number_input(
        "Alcool (% vol)",
        min_value=0.0,
        max_value=20.0,
        value=9.4,
        step=0.1,
        help="Teneur en alcool",
    )

st.divider()

# BOUTON DE PRÉDICTION
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 PRÉDIRE LA QUALITÉ", use_container_width=True)

# PRÉDICTION
if predict_button:
    with st.spinner("🔄 Analyse en cours..."):
        try:
            input_data = {
                "fixed_acidity": fixed_acidity,
                "volatile_acidity": volatile_acidity,
                "citric_acid": citric_acid,
                "residual_sugar": residual_sugar,
                "chlorides": chlorides,
                "free_sulfur_dioxide": free_sulfur_dioxide,
                "total_sulfur_dioxide": total_sulfur_dioxide,
                "density": density,
                "pH": pH,
                "sulphates": sulphates,
                "alcohol": alcohol,
            }

            response = requests.post(
                f"{st.session_state.api_url}/predict",
                json=input_data,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()

                if result.get("success"):
                    pred_dict = result["prediction"]
                    label = pred_dict["label"]
                    confidence = result.get("confidence", None)

                    # Ajout à l'historique
                    st.session_state.history.append({
                        "timestamp": datetime.now(),
                        "prediction": label,
                        "confidence": confidence if confidence is not None else 0.0,
                    })

                    st.success("✅ Prédiction réussie !")

                    st.markdown("---")
                    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])

                    with col_res2:
                        color_map = {
                            "Mauvais": "#ef4444",
                            "Moyen": "#eab308",
                            "Bon": "#22c55e",
                        }
                        color = color_map.get(label, "#666")

                        emoji_map = {
                            "Mauvais": "😞",
                            "Moyen": "😐",
                            "Bon": "😄",
                        }
                        emoji = emoji_map.get(label, "🍊")

                        desc_map = {
                            "Mauvais": "Qualité jugée faible.",
                            "Moyen": "Qualité correcte.",
                            "Bon": "Qualité élevée.",
                        }
                        description = desc_map.get(label, "Résultat de la prédiction.")

                        conf_str = f"{confidence*100:.1f}%" if confidence is not None else "N/A"

                        st.markdown(
                            f"""
                            <div style='text-align: center; padding: 2rem;
                                     background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
                                     border-radius: 15px; border: 2px solid {color};'>
                                <div style='font-size: 5rem; margin-bottom: 1rem;'>
                                    {emoji}
                                </div>
                                <h2 style='color: {color}; margin: 0;'>
                                    {label}
                                </h2>
                                <p style='color: #666; margin-top: 0.5rem;'>
                                    {description}
                                </p>
                                <p style='font-size: 1.2rem; font-weight: bold; color: {color};'>
                                    Confiance: {conf_str}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    
                else:
                    st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
            else:
                st.error(f"❌ Erreur API: Code {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API. Vérifiez que le serveur Flask est démarré.")
        except requests.exceptions.Timeout:
            st.error("⏱️ Délai d'attente dépassé. Le serveur met trop de temps à répondre.")
        except Exception as e:
            st.error(f"❌ Erreur inattendue: {str(e)}")



    

# FOOTER
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🍊 Application de Prédiction de Qualité de Jus</p>
        <p>Basée sur un modèle ML optimisé (XGBoost)</p>
    </div>
    """,
    unsafe_allow_html=True,
)
