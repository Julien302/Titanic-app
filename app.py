# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Données pour la partie Exploration/DataViz -------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")
df = load_data()

# ---------- Chargement du modèle déjà entraîné ----------------------
@st.cache_resource  # garde le modèle une fois en mémoire
def load_model():
    artifact = joblib.load("model.joblib")
    return artifact["model"], artifact["preprocessor"]

model, preprocessor = load_model()

# ---------- UI ------------------------------------------------------
st.title("Projet Titanic – version accélérée ⚡")
pages = ["Exploration", "DataVizualization", "Prédictions"]
page = st.sidebar.radio("Aller vers", pages)

if page == "Exploration":
    st.write(df.head(10))
    st.write(df.shape)
    st.dataframe(df.describe())
    if st.checkbox("Afficher les NA"):
        st.dataframe(df.isna().sum())

elif page == "DataVizualization":
    fig = plt.figure()
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x='Pclass', data=df)
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)
    
    fig = sns.displot(x='Age', data=df)
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=df)
    st.pyplot(fig)
    
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)
    
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)
    
    numeric_df = df.select_dtypes(include='number')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif page == "Prédictions":
    st.subheader("Tester le modèle enregistré")
    # mini-formulaire pour saisir les variables nécessaires
    sex = st.selectbox("Sexe", ["male", "female"])
    pclass = st.selectbox("Classe", [1, 2, 3])
    age = st.slider("Âge", 0, 80, 30)
    fare = st.number_input("Tarif (€)", 0.0, 600.0, 32.2)
    sibsp = st.number_input("Frères/Sœurs + conjoint·e", 0, 8, 0)
    parch = st.number_input("Parents + enfants", 0, 6, 0)
    embarked = st.selectbox("Embarquement", ["S", "C", "Q"])
    
    # transformer l'input utilisateur en DataFrame identique à X_train
    user_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "SibSp": sibsp,
        "Parch": parch,
        "Embarked": embarked
    }])
    
    if st.button("Prédire la survie"):
        # Utiliser le preprocesseur pour transformer les données
        user_processed = preprocessor.transform(user_df)
        proba = model.predict_proba(user_processed)[0][1]
        st.write(f"Probabilité de survie : **{proba:.1%}**")