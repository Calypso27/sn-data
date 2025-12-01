import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="1 – Exploration des Données", page_icon="📊")

@st.cache_data
def load_data():
    pages_dir = os.path.dirname(__file__)             
    project_dir = os.path.dirname(os.path.dirname(pages_dir)) 
    csv_path = os.path.join(project_dir, "juice_data.csv")

    return pd.read_csv(csv_path)

st.title("📊 Exploration des Données de Jus")

df = load_data()
st.subheader("Aperçu du jeu de données")
st.dataframe(df.head())

st.subheader("Dimensions et informations")
col1, col2 = st.columns(2)
with col1:
    st.write(f"Nombre de lignes : **{df.shape[0]}**")
    st.write(f"Nombre de colonnes : **{df.shape[1]}**")
with col2:
    st.write("Colonnes :")
    st.write(list(df.columns))

st.subheader("Statistiques descriptives")
st.dataframe(df.describe())

st.subheader("Distribution de la cible (quality_category)")
fig, ax = plt.subplots()
sns.countplot(data=df, x="quality_category", ax=ax)
ax.set_xlabel("Catégorie de qualité")
ax.set_ylabel("Nombre d'échantillons")
st.pyplot(fig)

st.subheader("Matrice de corrélation")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)
