# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string

# ------------------------ CONFIGURATION ------------------------
st.set_page_config(page_title="Analyse NLP Disneyland Paris", page_icon="🎢", layout="wide")

# ------------------------ STYLES CSS ------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #1f1f1f;
    }

    h1, h2, h3, h4 {
        font-weight: 600;
        color: #2c3e50;
    }

    .stButton>button {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        background-color: #4A90E2;
        color: white;
    }

    .stDataFrame, .stMetric {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------ FONCTION DE NETTOYAGE TEXTE ------------------------
def simple_clean(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# ------------------------ DONNÉES ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("avis_analytiques_disneyland.csv")
    if 'cleaned_review' not in df.columns:
        df['cleaned_review'] = df['review_format'].apply(simple_clean)
    return df

df = load_data()

# ------------------------ SIDEBAR ------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/fr/thumb/e/ec/Logo_Disneyland_Paris.svg/1280px-Logo_Disneyland_Paris.svg.png", use_container_width=True)
st.sidebar.title("🧭 Navigation")
onglet = st.sidebar.radio("Aller à :", [
    "🏠 Synthèse", 
    "🧠 Thèmes", 
    "💬 Sentiments", 
    "📊 Filtres personnalisés",
    "📈 Timeline & WordCloud"
])

# ------------------------ PAGE : Synthèse ------------------------
if "Synthèse" in onglet:
    st.title("🧠 Analyse NLP des Avis Disneyland Paris")

    st.markdown("""
    ---
    ### 🎯 Objectif
    Comprendre les retours clients grâce à une analyse automatique :
    - des **sentiments** exprimés
    - des **thèmes** évoqués
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📌 Nombre d'avis", len(df))
    with col2:
        moyenne = round(df['stars'].mean(), 2)
        st.metric("⭐ Note moyenne", f"{moyenne} / 5")

    st.markdown("🔍 Cette analyse s’appuie sur BERT pour la détection des sentiments et LDA pour l'extraction de thèmes.")

# ------------------------ PAGE : Thèmes ------------------------
elif "Thèmes" in onglet:
    st.title("🧠 Thèmes Dominants (LDA)")

    topics = sorted(df['topic_dominant'].dropna().unique())
    selected = st.selectbox("🎯 Choisissez un thème", topics)
    filtered = df[df['topic_dominant'] == selected]

    st.success(f"{len(filtered)} avis associés au thème **{selected}**")
    st.dataframe(filtered[['review_format', 'model_prediction', 'stars']], use_container_width=True)

    fig = px.histogram(filtered, x="stars", nbins=5, color_discrete_sequence=["#636EFA"],
                       title="Distribution des notes pour ce thème")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------ PAGE : Sentiments ------------------------
elif "Sentiments" in onglet:
    st.title("💬 Analyse des Sentiments")

    sentiment_counts = df['model_prediction'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Nombre']

    fig = px.bar(sentiment_counts, x='Sentiment', y='Nombre',
                 color='Sentiment', title="Répartition des Sentiments",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Note moyenne", f"{round(df['stars'].mean(), 2)} / 5")
    with col2:
        st.metric("💬 Total d'avis", len(df))

# ------------------------ PAGE : Filtres Personnalisés ------------------------
elif "Filtres" in onglet:
    st.title("📊 Filtres Personnalisés")
    st.markdown("Affinez les avis affichés selon vos critères :")

    with st.expander("🎛️ Filtres avancés"):
        col1, col2 = st.columns(2)
        with col1:
            stars_range = st.slider("⭐ Filtrer par note", 1, 5, (1, 5))
        with col2:
            sentiments = st.multiselect("💬 Sentiment", options=sorted(df['model_prediction'].dropna().unique()))

    filtered_df = df[
        (df['stars'] >= stars_range[0]) & (df['stars'] <= stars_range[1])
    ]
    if sentiments:
        filtered_df = filtered_df[filtered_df['model_prediction'].isin(sentiments)]

    st.info(f"🔎 {len(filtered_df)} avis correspondent à vos critères.")
    st.dataframe(filtered_df[['review_format', 'stars', 'model_prediction', 'topic_dominant']], use_container_width=True)

    fig = px.histogram(filtered_df, x="stars", color="model_prediction", barmode="group",
                       title="Répartition des notes selon les sentiments",
                       color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Télécharger ces avis", filtered_df.to_csv(index=False),
                       file_name="avis_filtrés.csv", mime="text/csv")

# ------------------------ PAGE : Timeline & WordCloud ------------------------
elif "Timeline" in onglet:
    st.title("📈 Évolution des Sentiments & Nuage de Mots")

    # Timeline
    if 'model_prediction' in df.columns and 'month_year' in df.columns:
        df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')
        timeline_df = df.dropna(subset=['month_year'])

        sentiments_over_time = timeline_df.groupby([
            timeline_df['month_year'].dt.to_period('M'), 'model_prediction']) \
            .size().reset_index(name='count')
        sentiments_over_time['month_year'] = sentiments_over_time['month_year'].astype(str)

        fig = px.line(sentiments_over_time, x="month_year", y="count", color="model_prediction",
                      title="Évolution des Sentiments dans le Temps",
                      markers=True, line_shape='spline')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Les colonnes 'model_prediction' ou 'month_year' sont manquantes.")

    # WordCloud
    st.subheader("☁️ Nuage de mots par sentiment")

    sentiments = df['model_prediction'].dropna().unique().tolist()
    selected_sentiment = st.selectbox("Choisir un sentiment", sentiments)

    text = " ".join(df[df['model_prediction'] == selected_sentiment]['cleaned_review'].dropna().astype(str))

    if text:
        wordcloud = WordCloud(width=1000, height=400, background_color='white',
                              max_words=100, collocations=False).generate(text)

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Aucun texte disponible pour ce sentiment.")