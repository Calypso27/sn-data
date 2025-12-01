import streamlit as st
import requests

st.set_page_config(page_title="5 – Test de l'API", page_icon="🛰️")

st.title("🛰️ Test de l'API de Prédiction (Hugging Face)")

API_URL = "https://calypso-mb-api-j.hf.space"

st.sidebar.header("🔗 Informations API")
st.sidebar.write(f"**URL :** {API_URL}")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.sidebar.success("✅ API connectée")
        st.sidebar.write(f"Statut: {health_data.get('status', 'N/A')}")
    else:
        st.sidebar.warning(f"⚠️ Statut: {health_response.status_code}")
except Exception as e:
    st.sidebar.error(f"❌ Erreur connexion: {e}")

FEATURES = {
    "fixed_acidity": "Acidité fixe (g/L)",
    "volatile_acidity": "Acidité volatile (g/L)",
    "citric_acid": "Acide citrique (g/L)",
    "residual_sugar": "Sucre résiduel (g/L)",
    "chlorides": "Chlorures (g/L)",
    "free_sulfur_dioxide": "SO2 libre (mg/L)",
    "total_sulfur_dioxide": "SO2 total (mg/L)",
    "density": "Densité (g/cm³)",
    "pH": "pH",
    "sulphates": "Sulfates (g/L)",
    "alcohol": "Alcool (% vol)",
}

DEFAULT_VALUES = {
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

with st.form("api_form"):
    st.subheader("🍊 Caractéristiques du Jus")

    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        for feature in list(FEATURES.keys())[:6]:
            inputs[feature] = st.number_input(
                FEATURES[feature],
                value=DEFAULT_VALUES[feature],
                format="%.4f",
                key=feature,
            )
    with col2:
        for feature in list(FEATURES.keys())[6:]:
            inputs[feature] = st.number_input(
                FEATURES[feature],
                value=DEFAULT_VALUES[feature],
                format="%.4f",
                key=feature,
            )

    submit = st.form_submit_button("🚀 Appeler l'API")

if submit:
    with st.spinner("🔮 Interrogation de l'API..."):
        try:
            payload = {k: float(v) for k, v in inputs.items()}

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    pred = result["prediction"]
                    raw = pred.get("raw")
                    label = pred.get("label")
                    confidence = result.get("confidence")

                    quality_info = {
                        0: {"emoji": "😞", "text": "Mauvaise qualité", "color": "red"},
                        1: {"emoji": "😐", "text": "Qualité moyenne", "color": "orange"},
                        2: {"emoji": "😊", "text": "Bonne qualité", "color": "green"},
                    }.get(int(raw) if raw is not None else -1, {"emoji": "❓", "text": label, "color": "gray"})

                    st.success("✅ Prédiction réussie !")

                    st.markdown(
                        f"""
                        <div style='border: 2px solid {quality_info['color']}; padding: 1rem; border-radius: 10px; text-align: center;'>
                            <div style='font-size: 3rem;'>{quality_info['emoji']}</div>
                            <h3 style='color: {quality_info['color']};'>{quality_info['text']}</h3>
                            <p>Label API : <b>{label}</b> (classe {raw})</p>
                            <p>Confiance : <b>{confidence*100:.1f}%</b> si disponible</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"❌ Erreur retournée par l'API : {result.get('error', 'Inconnue')}")
            else:
                st.error(f"❌ Erreur HTTP : code {response.status_code}")

        except Exception as e:
            st.error(f"❌ Erreur lors de l'appel API : {e}")
