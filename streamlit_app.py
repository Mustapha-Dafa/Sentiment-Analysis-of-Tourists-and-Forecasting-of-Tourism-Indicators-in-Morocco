import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO
import time
from streamlit_option_menu import option_menu
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

VARIABLES_CONFIG = {
    "TES": {
        "order": (0, 1, 0), 
        "seasonal_order": (2, 2, 1, 12)
    },
    "MRE": {  
        "order": (0, 1, 1),
        "seasonal_order": (1, 1, 1, 12)
    },
    "Nuité": {
        "order": (2, 1, 2),
        "seasonal_order": (1, 1, 1, 12)
    }
}


# Obtenir le répertoire du script
BASE_DIR = Path(__file__).parent

# Définir les chemins relatifs
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

REVIEWS_FILE = DATA_DIR / "reviews" / "reviews_classed.csv"
KPI_FILE = DATA_DIR / "tourism" / "KPI_touristique.csv"
NUITES_FILE = DATA_DIR / "tourism" / "Nuite_par_destination.csv"
LOGO_FILE = IMAGES_DIR / "logo.png"



# calcule des prévisions 
def predict_simple(df , target , horizon ):
    train = df[target]
    model = SARIMAX(train,order=VARIABLES_CONFIG[target]["order"],seasonal_order=VARIABLES_CONFIG[target]["seasonal_order"])
    results = model.fit(disp=False)
    preds= results.forecast(horizon)
    predictions = pd.DataFrame(preds)
    return predictions

def predict_progressive(df,target , horizon):
    train = df[target]
    predictions = pd.DataFrame(columns=[target])
    for i in range(horizon):
        model = SARIMAX(train,order=VARIABLES_CONFIG[target]["order"],seasonal_order=VARIABLES_CONFIG[target]["seasonal_order"])
        results = model.fit(disp=False)
        pred = results.forecast(1)
        new_row = pd.DataFrame({target: [pred.iloc[0]]}, index=[pred.index[0]])
        train =pd.concat([train,new_row])
        predictions =pd.concat([predictions,new_row])
    return predictions





# Configuration de la page
st.set_page_config(
    page_title="Analyse Tourisme Maroc",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)
if LOGO_FILE.exists():
    st.sidebar.image(str(LOGO_FILE))
else:
    st.sidebar.warning("⚠️ Logo non trouvé")


# Fonction de chargement des données
@st.cache_data
def load_data():
    """Charge et prétraite les données"""
    try:
        # Vérifier l'existence des fichiers
        if not REVIEWS_FILE.exists():
            st.error(f"❌ Fichier introuvable : {REVIEWS_FILE}")
            return None, None, None, None
        
        if not KPI_FILE.exists():
            st.error(f"❌ Fichier introuvable : {KPI_FILE}")
            return None, None, None, None
        
        if not NUITES_FILE.exists():
            st.error(f"❌ Fichier introuvable : {NUITES_FILE}")
            return None, None, None, None
        
        # Charger les données des avis
        df_reviews = pd.read_csv(REVIEWS_FILE)
        df_reviews['date'] = pd.to_datetime(df_reviews['date'])
        
        # Charger les données des arrivées
        df_arrivals = pd.read_csv(KPI_FILE)
        df = df_arrivals.copy()
        df_arrivals['date'] = pd.to_datetime(df_arrivals['date'])
        
        # Charger nuitée par destination 
        df_nuites_ville = pd.read_csv(NUITES_FILE)
        df_nuites_ville['date'] = pd.to_datetime(df_nuites_ville['date'])

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.asfreq('MS')  # MS = Month Start
        df = df.sort_index()
        
        return df_reviews, df_arrivals, df, df_nuites_ville
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None, None, None, None

# Fonction pour créer un nuage de mots
def create_wordcloud(text_data, sentiment=None):
    """Crée un nuage de mots à partir du texte"""
    if sentiment:
        text = ' '.join(text_data[text_data['sentiment'] == sentiment]['text'].dropna())
    else:
        text = ' '.join(text_data['text'].dropna())
    
    if text.strip():
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    return None

# Fonction pour calculer les statistiques de base
def calculate_stats(df_reviews):
    """Calcule les statistiques descriptives"""
    # Vérifier si des notes sont disponibles
    notes_disponibles = df_reviews['note'].notna().sum() > 0
    note_moyenne = df_reviews['note'].mean() if notes_disponibles else None
    
    stats = {
        'total_avis': len(df_reviews),
        'note_moyenne': note_moyenne,
        'notes_disponibles': notes_disponibles,
        'villes': df_reviews['city'].nunique(),
        'plateformes': df_reviews['plateforme'].nunique(),
        'categories': df_reviews['category'].nunique(),
        'sentiment_positif_pct': (df_reviews['sentiment'] == 'positif').sum() / len(df_reviews) * 100 if len(df_reviews) > 0 else 0,
        'sentiment_négatif_pct': (df_reviews['sentiment'] == 'négatif').sum() / len(df_reviews) * 100 if len(df_reviews) > 0 else 0,
        'sentiment_neutre_pct': (df_reviews['sentiment'] == 'neutre').sum() / len(df_reviews) * 100 if len(df_reviews) > 0 else 0,
    }
    return stats

# Chargement des données
df_reviews, df_arrivals ,df ,df_nuites_ville= load_data()

if df_reviews is not None and df_arrivals is not None:
    
    with st.sidebar:
        page = option_menu(None, ["Dashboard", "Analyse des Sentiments", " Analyse des Arrivées",
             "Analyse des Nuitées","Générer les prévisions","Exploration les Avis"], 
        icons=['house', 'emoji-smile','graph-up','building' ,'graph-up-arrow','chat-dots'], 
        menu_icon="cast", default_index=0 )#, orientation="horizontal")

    # Sidebar - Filtres
    st.sidebar.title(" Filtres")
    
    # Filtre par ville
    st.sidebar.subheader("Villes")
    all_cities = ['Toutes'] + sorted(df_reviews['city'].unique().tolist())
    selected_cities = st.sidebar.multiselect(
        "Sélectionner les villes",
        options=all_cities,
        default=['Toutes']
    )
    
    # Filtre par plateforme
    st.sidebar.subheader("Plateformes")
    all_platforms = ['Toutes'] + sorted(df_reviews['plateforme'].unique().tolist())
    selected_platforms = st.sidebar.multiselect(
        "Sélectionner les plateformes",
        options=all_platforms,
        default=['Toutes']
    )
    
    # Filtre par catégorie
    st.sidebar.subheader("Catégories")
    all_categories = ['Toutes'] + sorted(df_reviews['category'].unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Sélectionner les catégories",
        options=all_categories,
        default=['Toutes']
    )
    
    # Filtre par sentiment
    st.sidebar.subheader("Sentiments")
    all_sentiments = ['Tous'] + sorted(df_reviews['sentiment'].unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Sélectionner les sentiments",
        options=all_sentiments,
        default=['Tous']
    )
    
    # Application des filtres
    df_filtered = df_reviews.copy()
    
    
    # Filtre par ville
    if 'Toutes' not in selected_cities and selected_cities:
        df_filtered = df_filtered[df_filtered['city'].isin(selected_cities)]
    
    # Filtre par plateforme
    if 'Toutes' not in selected_platforms and selected_platforms:
        df_filtered = df_filtered[df_filtered['plateforme'].isin(selected_platforms)]
    
    # Filtre par catégorie
    if 'Toutes' not in selected_categories and selected_categories:
        df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]
    
    # Filtre par sentiment
    if 'Tous' not in selected_sentiments and selected_sentiments:
        df_filtered = df_filtered[df_filtered['sentiment'].isin(selected_sentiments)]
    
    # Navigation
    #st.sidebar.markdown("---")
    

    # ==================== PAGE DASHBOARD ====================
    if page == "Dashboard":
        st.title("Dashboard ")
        st.markdown("---")
        
        # Calculer les statistiques
        stats = calculate_stats(df_filtered)
        
        # KPIs principaux
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Avis", f"{stats['total_avis']:,}")
        with col2:
            if stats['notes_disponibles']:
                st.metric("⭐ Note Moyenne", f"{stats['note_moyenne']:.2f}/5")
            else:
                st.metric("⭐ Note Moyenne", "N/A")
        with col3:
            st.metric("Positifs", f"{stats['sentiment_positif_pct']:.1f}%")
        with col4:
            st.metric("Villes", stats['villes'])
        with col5:
            st.metric("Plateformes", stats['plateformes'])
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des sentiments
            st.subheader("Distribution des Sentiments")
            sentiment_counts = df_filtered['sentiment'].value_counts()
            colors = {'positif': '#2ecc71', 'neutre': '#f39c12', 'négatif': '#e74c3c'}
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution des notes
            st.subheader("Distribution des Notes")
            if df_filtered['note'].notna().sum() > 0:
                note_counts = df_filtered['note'].dropna().value_counts().sort_index()
                fig = px.bar(
                    x=note_counts.index,
                    y=note_counts.values,
                    labels={'x': 'Note', 'y': 'Nombre d\'avis'},
                    color=note_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Aucune note disponible pour cette sélection (ex: TripAdvisor)")
        
        # Évolution temporelle
        st.subheader("Évolution Temporelle des Sentiments")
        df_temp = df_filtered.groupby([df_filtered['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
        df_temp.index = df_temp.index.to_timestamp()
        
        fig = go.Figure()
        for sentiment in df_temp.columns:
            color = colors.get(sentiment, '#95a5a6')
            fig.add_trace(go.Scatter(
                x=df_temp.index,
                y=df_temp[sentiment],
                name=sentiment.capitalize(),
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Nombre d'avis",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top villes et catégories
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Villes")
            city_counts = df_filtered['city'].value_counts().head(10)
            fig = px.bar(
                x=city_counts.values,
                y=city_counts.index,
                orientation='h',
                labels={'x': 'Nombre d\'avis', 'y': 'Ville'},
                color=city_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Catégories")
            category_counts = df_filtered['category'].value_counts().head(10)
            fig = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                labels={'x': 'Nombre d\'avis', 'y': 'Catégorie'},
                color=category_counts.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== PAGE ANALYSE SENTIMENTS ====================
    elif page == "Analyse des Sentiments":
        st.title("Analyse Détaillée des Sentiments")
        st.markdown("---")
        
        # Sentiments par ville
        st.subheader("Sentiments par Ville")
        df_city_sent = df_filtered.groupby(['city', 'sentiment']).size().unstack(fill_value=0)
        df_city_sent = df_city_sent.sort_values(by='positif', ascending=False).head(15)
        
        fig = go.Figure()
        colors_sentiment = {'positif': '#2ecc71', 'neutre': '#f39c12', 'négatif': '#e74c3c'}
        
        for sentiment in df_city_sent.columns:
            fig.add_trace(go.Bar(
                name=sentiment.capitalize(),
                x=df_city_sent.index,
                y=df_city_sent[sentiment],
                marker_color=colors_sentiment.get(sentiment, '#95a5a6')
            ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="Ville",
            yaxis_title="Nombre d'avis",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiments par plateforme
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiments par Plateforme")
            df_platform_sent = df_filtered.groupby(['plateforme', 'sentiment']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            for sentiment in df_platform_sent.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.capitalize(),
                    x=df_platform_sent.index,
                    y=df_platform_sent[sentiment],
                    marker_color=colors_sentiment.get(sentiment, '#95a5a6')
                ))
            
            fig.update_layout(barmode='stack', xaxis_title="Plateforme", yaxis_title="Nombre d'avis")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiments par Catégorie")
            df_cat_sent = df_filtered.groupby(['category', 'sentiment']).size().unstack(fill_value=0)
            df_cat_sent = df_cat_sent.sort_values(by='positif', ascending=False).head(10)
            
            fig = go.Figure()
            for sentiment in df_cat_sent.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.capitalize(),
                    y=df_cat_sent.index,
                    x=df_cat_sent[sentiment],
                    orientation='h',
                    marker_color=colors_sentiment.get(sentiment, '#95a5a6')
                ))
            
            fig.update_layout(barmode='stack', xaxis_title="Nombre d'avis", yaxis_title="Catégorie")
            st.plotly_chart(fig, use_container_width=True)
        

        st.markdown("---")
        st.title("Statistiques Détaillées")
        
        
        tab1, tab2, tab3, tab4 = st.tabs(["Par Ville", "Par Plateforme", "Par Catégorie", "Temporelles"])
        
        with tab1:
            st.subheader("Analyse Détaillée par Ville")
            
            # Vérifier si des notes sont disponibles
            notes_disponibles = df_filtered['note'].notna().sum() > 0
            
            if notes_disponibles:
                city_stats = df_filtered.groupby('city').agg({
                    'note': ['mean', 'count'],
                    'sentiment': lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)
                city_stats.columns = ['Note Moyenne', 'Nombre d\'Avis', 'Sentiment Positif (%)']
            else:
                city_stats = df_filtered.groupby('city').agg({
                    'city': 'count',
                    'sentiment': lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)
                city_stats.columns = ['Nombre d\'Avis', 'Sentiment Positif (%)']
            
            city_stats = city_stats.sort_values('Nombre d\'Avis', ascending=False)
            
            st.dataframe(city_stats, use_container_width=True)
            
            # Graphique
            top_cities = city_stats.head(15)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=top_cities.index,
                    y=top_cities['Nombre d\'Avis'],
                    name='Nombre d\'Avis',
                    marker_color='#3498db'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=top_cities.index,
                    y=top_cities['Sentiment Positif (%)'],
                    name='Sentiment Positif (%)',
                    line=dict(color='#2ecc71', width=3),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Ville")
            fig.update_yaxes(title_text="Nombre d'Avis", secondary_y=False)
            fig.update_yaxes(title_text="Sentiment Positif (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Analyse Détaillée par Plateforme")
            
            # Vérifier si des notes sont disponibles
            notes_disponibles = df_filtered['note'].notna().sum() > 0
            
            if notes_disponibles:
                platform_stats = df_filtered.groupby('plateforme').agg({
                    'note': ['mean', 'count'],
                    'sentiment': lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)
                platform_stats.columns = ['Note Moyenne', 'Nombre d\'Avis', 'Sentiment Positif (%)']
            else:
                platform_stats = df_filtered.groupby('plateforme').agg({
                    'plateforme': 'count',
                    'sentiment': lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)
                platform_stats.columns = ['Nombre d\'Avis', 'Sentiment Positif (%)']
            
            platform_stats = platform_stats.sort_values('Nombre d\'Avis', ascending=False)
            
            st.dataframe(platform_stats, use_container_width=True)
            
            # Graphique comparatif
            if notes_disponibles:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Note Moyenne',
                    x=platform_stats.index,
                    y=platform_stats['Note Moyenne'],
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    xaxis_title="Plateforme",
                    yaxis_title="Note Moyenne",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Afficher le sentiment positif à la place
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Sentiment Positif',
                    x=platform_stats.index,
                    y=platform_stats['Sentiment Positif (%)'],
                    marker_color='#2ecc71'
                ))
                
                fig.update_layout(
                    xaxis_title="Plateforme",
                    yaxis_title="Sentiment Positif (%)",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Analyse Détaillée par Catégorie")
            
            # Vérifier si des notes sont disponibles
            notes_disponibles = df_filtered['note'].notna().sum() > 0
            
            if notes_disponibles:
                category_stats = df_filtered.groupby('category').agg({
                    'note': ['mean', 'count'],
                    'sentiment': lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)
                category_stats.columns = ['Note Moyenne', 'Nombre d\'Avis', 'Sentiment Positif (%)']
            else:
                category_stats = df_filtered.groupby('category').agg({
                    'category': 'count',
                    'sentiment': lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0
                }).round(2)
                category_stats.columns = ['Nombre d\'Avis', 'Sentiment Positif (%)']
            
            category_stats = category_stats.sort_values('Nombre d\'Avis', ascending=False)
            
            st.dataframe(category_stats, use_container_width=True)
            
            # Top catégories
            top_categories = category_stats.head(10)
            
            fig = px.bar(
                top_categories,
                y=top_categories.index,
                x='Nombre d\'Avis',
                orientation='h',
                color='Sentiment Positif (%)',
                color_continuous_scale='RdYlGn',
                labels={'Nombre d\'Avis': 'Nombre d\'Avis', 'Sentiment Positif (%)': 'Sentiment Positif (%)'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Analyse Temporelle")
            
            # Vérifier si des notes sont disponibles
            notes_disponibles = df_filtered['note'].notna().sum() > 0
            
            # Par année
            df_filtered['year'] = df_filtered['date'].dt.year
            
            if notes_disponibles:
                yearly_stats = df_filtered.groupby('year').agg({
                    'note': 'mean',
                    'sentiment': ['count', lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0]
                }).round(2)
                yearly_stats.columns = ['Note Moyenne', 'Nombre d\'Avis', 'Sentiment Positif (%)']
            else:
                yearly_stats = df_filtered.groupby('year').agg({
                    'sentiment': ['count', lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0]
                }).round(2)
                yearly_stats.columns = ['Nombre d\'Avis', 'Sentiment Positif (%)']
            
            st.subheader("Statistiques Annuelles")
            st.dataframe(yearly_stats, use_container_width=True)
            
            # Par mois
            df_filtered['month'] = df_filtered['date'].dt.month
            
            if notes_disponibles:
                monthly_stats = df_filtered.groupby('month').agg({
                    'note': 'mean',
                    'sentiment': ['count', lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0]
                }).round(2)
                monthly_stats.columns = ['Note Moyenne', 'Nombre d\'Avis', 'Sentiment Positif (%)']
            else:
                monthly_stats = df_filtered.groupby('month').agg({
                    'sentiment': ['count', lambda x: (x == 'positif').sum() / len(x) * 100 if len(x) > 0 else 0]
                }).round(2)
                monthly_stats.columns = ['Nombre d\'Avis', 'Sentiment Positif (%)']
            
            month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                          'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
            monthly_stats.index = [month_names[i-1] for i in monthly_stats.index]
            
            st.subheader("Saisonnalité Mensuelle")
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=monthly_stats.index,
                    y=monthly_stats['Nombre d\'Avis'],
                    name='Nombre d\'Avis',
                    marker_color='#3498db'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_stats.index,
                    y=monthly_stats['Sentiment Positif (%)'],
                    name='Sentiment Positif (%)',
                    line=dict(color='#2ecc71', width=3),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Mois")
            fig.update_yaxes(title_text="Nombre d'Avis", secondary_y=False)
            fig.update_yaxes(title_text="Sentiment Positif (%)", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)

    
    # ==================== PAGE ANALYSE ARRIVÉES ====================
    elif page == " Analyse des Arrivées":
        st.title(" Analyse des Arrivées Touristiques")
        st.markdown("---")
        
        # KPIs des arrivées
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_arrivals = df_arrivals['Total Arrivées'].sum()
            st.metric("Total Arrivées", f"{total_arrivals:,.0f}")
        with col2:
            avg_tes = df_arrivals['TES'].mean()
            st.metric("TES Moyen", f"{avg_tes:,.0f}")
        with col3:
            avg_mre = df_arrivals['MRE'].mean()
            st.metric("MRE Moyen", f"{avg_mre:,.0f}")
        with col4:
            avg_occup = df_arrivals['Taux d\'occupation'].mean()
            st.metric("Taux d'occupation Moyen", f"{avg_occup:.1f}%")
        
        # Évolution des arrivées
        st.subheader("Évolution des Arrivées Touristiques")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_arrivals['date'],
            y=df_arrivals['Total Arrivées'],
            name='Total Arrivées',
            line=dict(color='#3498db', width=3),
            fill='tozeroy'
        ))
        fig.add_trace(go.Scatter(
            x=df_arrivals['date'],
            y=df_arrivals['TES'],
            name='TES',
            line=dict(color='#e74c3c', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_arrivals['date'],
            y=df_arrivals['MRE'],
            name='MRE',
            line=dict(color='#2ecc71', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Nombre d'arrivées",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition TES vs MRE
            st.subheader("Répartition TES vs MRE")
            total_tes = df_arrivals['TES'].sum()
            total_mre = df_arrivals['MRE'].sum()
            
            fig = px.pie(
                values=[total_tes, total_mre],
                names=['TES', 'MRE'],
                color_discrete_sequence=['#e74c3c', '#2ecc71'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Évolution du taux d'occupation
            st.subheader("Évolution du Taux d'Occupation")
            fig = px.line(
                df_arrivals,
                x='date',
                y='Taux d\'occupation',
                markers=True
            )
            fig.add_hline(
                y=df_arrivals['Taux d\'occupation'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text="Moyenne"
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Taux d'occupation (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des nuitées
        st.subheader("Analyse des Nuitées")
        fig = px.area(
            df_arrivals,
            x='date',
            y='Nuité',
            labels={'Nuité': 'Nombre de nuitées'},
            color_discrete_sequence=['#9b59b6']
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Nombre de nuitées")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques mensuelles
        st.subheader("Statistiques Mensuelles")
        df_arrivals['month'] = df_arrivals['date'].dt.month
        monthly_stats = df_arrivals.groupby('month').agg({
            'Total Arrivées': 'mean',
            'TES': 'mean',
            'MRE': 'mean',
            'Taux d\'occupation': 'mean'
        }).round(2)
        
        month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                       'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        monthly_stats.index = [month_names[i-1] for i in monthly_stats.index]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_stats.index,
            y=monthly_stats['Total Arrivées'],
            name='Arrivées Moyennes',
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            xaxis_title="Mois",
            yaxis_title="Nombre moyen d'arrivées",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # NOUVELLE page : ANALYSE DES NUITÉES PAR VILLE
    # ═══════════════════════════════════════════════════════════════════
    elif page =="Analyse des Nuitées"  : 
            st.title("Analyse des Nuitées ")
            st.markdown("---")
            
            # Charger les données de nuitées par ville
            try:
                
                # KPIs des nuitées par destination
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_nuites = df_nuites_ville['nuité'].sum()
                    st.metric("Total Nuitées", f"{total_nuites:,.0f}")
                with col2:
                    nb_destinations = df_nuites_ville['destination'].nunique()
                    st.metric("Destinations", nb_destinations)
                with col3:
                    moyenne_nuites = df_nuites_ville['nuité'].mean()
                    st.metric("Moyenne Nuitées", f"{moyenne_nuites:,.0f}")
                with col4:
                    destination_top = df_nuites_ville.groupby('destination')['nuité'].sum().idxmax()
                    st.metric("Top Destination", destination_top)
                
                st.markdown("---")
                
                # Graphiques en colonnes
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top 10 destinations par nuitées totales
                    st.subheader("Top 10 Destinations par Nuitées")
                    top_destinations = df_nuites_ville.groupby('destination')['nuité'].sum().sort_values(ascending=False).head(10)
                    
                    fig = px.bar(
                        x=top_destinations.values,
                        y=top_destinations.index,
                        orientation='h',
                        labels={'x': 'Nombre de nuitées', 'y': 'Destination'},
                        color=top_destinations.values,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Répartition des nuitées par destination (Top 10)
                    st.subheader("Répartition des Nuitées (Top 10)")
                    
                    fig = px.pie(
                        values=top_destinations.values,
                        names=top_destinations.index,
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Évolution temporelle des nuitées par destination (Top 5)
                st.subheader("Évolution Temporelle des Nuitées (Top 5 Destinations)")
                
                # Sélectionner les 5 destinations avec le plus de nuitées
                top5_destinations = df_nuites_ville.groupby('destination')['nuité'].sum().sort_values(ascending=False).head(5).index
                df_top5 = df_nuites_ville[df_nuites_ville['destination'].isin(top5_destinations)]
                
                fig = px.line(
                    df_top5,
                    x='date',
                    y='nuité',
                    color='destination',
                    markers=True,
                    labels={'nuité': 'Nombre de nuitées', 'date': 'Date', 'destination': 'Destination'}
                )
                fig.update_layout(
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
    
                # Préparer les données pour la heatmap
                df_nuites_ville['mois'] = df_nuites_ville['date'].dt.month
                df_nuites_ville['annee'] = df_nuites_ville['date'].dt.year
                
                # Créer une pivot table pour les 15 meilleures destinations
                
                
                # Renommer les colonnes pour afficher les noms de mois
                month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                            'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
                
                
                # Statistiques détaillées par destination
                st.markdown("---")
                st.subheader("Statistiques Détaillées par Destination")
                
                # Calculer les statistiques
                stats_destinations = df_nuites_ville.groupby('destination').agg({
                    'nuité': ['sum', 'mean', 'max', 'min', 'count']
                }).round(0)
                stats_destinations.columns = ['Total', 'Moyenne', 'Maximum', 'Minimum', 'Nb Périodes']
                stats_destinations = stats_destinations.sort_values('Total', ascending=False)
                
                # Ajouter une colonne de pourcentage
                stats_destinations['% du Total'] = (stats_destinations['Total'] / stats_destinations['Total'].sum() * 100).round(2)
                
                st.dataframe(
                    stats_destinations,
                    use_container_width=True,
                    column_config={
                        "Total": st.column_config.NumberColumn(format="%.0f"),
                        "Moyenne": st.column_config.NumberColumn(format="%.0f"),
                        "Maximum": st.column_config.NumberColumn(format="%.0f"),
                        "Minimum": st.column_config.NumberColumn(format="%.0f"),
                        "% du Total": st.column_config.NumberColumn(format="%.2f%%")
                    }
                )
                
                # Analyse de saisonnalité
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Saisonnalité Moyenne par Mois")
                    
                    monthly_avg = df_nuites_ville.groupby('mois')['nuité'].mean().sort_index()
                    monthly_avg.index = [month_names[int(m)-1] for m in monthly_avg.index]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=monthly_avg.index,
                        y=monthly_avg.values,
                        marker_color='#3498db',
                        name='Moyenne mensuelle'
                    ))
                    
                    # Ajouter une ligne de tendance
                    fig.add_trace(go.Scatter(
                        x=monthly_avg.index,
                        y=monthly_avg.values,
                        mode='lines',
                        line=dict(color='#e74c3c', width=3),
                        name='Tendance'
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Mois",
                        yaxis_title="Nuitées moyennes",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Évolution Annuelle Totale")
                    
                    yearly_total = df_nuites_ville.groupby('annee')['nuité'].sum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=yearly_total.index,
                        y=yearly_total.values,
                        marker_color='#2ecc71',
                        name='Total annuel'
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Année",
                        yaxis_title="Total nuitées",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Filtre interactif par destination
                st.markdown("---")
                st.subheader("Analyse Détaillée par Destination")
                
                selected_destination = st.selectbox(
                    "Sélectionnez une destination",
                    options=sorted(df_nuites_ville['destination'].unique())
                )
                
                if selected_destination:
                    df_selected = df_nuites_ville[df_nuites_ville['destination'] == selected_destination]
                    
                    # KPIs pour la destination sélectionnée
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_dest = df_selected['nuité'].sum()
                        st.metric("Total Nuitées", f"{total_dest:,.0f}")
                    with col2:
                        moyenne_dest = df_selected['nuité'].mean()
                        st.metric("Moyenne", f"{moyenne_dest:,.0f}")
                    with col3:
                        max_dest = df_selected['nuité'].max()
                        st.metric("Maximum", f"{max_dest:,.0f}")
                    with col4:
                        part_dest = (total_dest / total_nuites * 100)
                        st.metric("Part du Total", f"{part_dest:.2f}%")
                    
                    # Graphique d'évolution pour la destination
                    fig = px.area(
                        df_selected,
                        x='date',
                        y='nuité',
                        labels={'nuité': 'Nombre de nuitées', 'date': 'Date'},
                        color_discrete_sequence=['#9b59b6']
                    )
                    fig.update_layout(
                        title=f"Évolution des nuitées - {selected_destination}",
                        xaxis_title="Date",
                        yaxis_title="Nombre de nuitées"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                                
            
            except FileNotFoundError:
                st.error("⚠️ Fichier des nuitées par destination non trouvé. Veuillez vérifier le chemin du fichier.")
                st.info("📁 Le fichier doit contenir les colonnes : date, nuité, destination")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement des données : {e}")



    # ==================== PAGE Générer les prévisions ====================
    elif page == "Générer les prévisions" :
        st.title("Module de Prévisions")
        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════
        # SECTION CONFIGURATION - Dans la page principale
        # ═══════════════════════════════════════════════════════════════

        st.header("⚙️ Configuration des prévisions")

        # Création de colonnes pour organiser les paramètres
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Variable à prédire")
            st.markdown("""
                <style>
                div[data-baseweb="select"] > div {
                    height: 45px;               /* hauteur du select */
                    font-size: 15px;              /* taille du texte */
                }
                div[data-baseweb="select"] span {
                    font-size: 15px !important;   /* taille du texte interne */
                }
                </style>
            """, unsafe_allow_html=True)
            target_variable = st.selectbox(
                "Choisissez la variable",
                options=["TES", "MRE", "Total Arrivées", "Nuité"],
                format_func=lambda x: {
                    "TES": "Arrivées de touristes étrangers des séjour (TES)",
                    "MRE": "Arrivées des marocains résidents à l'étranger (MRE)",
                    "Total Arrivées": "Arrivées des touristes (Total)",
                    "Nuité": "Nuitées"
                }[x],
                help="Sélectionnez la variable que vous souhaitez prédire"
            )
            
            st.info(f" Le modèle a été entraîné sur des données : Jan 2010 → Mai 2025")

        with col2:
            st.subheader("Paramètres temporels")
            horizon = st.slider(
                "Horizon de prédiction (mois)",
                min_value=1,
                max_value=24,
                value=12,
                help="Nombre de mois à prédire dans le futur"
            )
            
            start_date = st.date_input(
                "Date de début des prédictions",
                value=pd.Timestamp("2025-05-01"),
                help="Première période à prédire"
            )

        with col3:
            st.subheader("Mode de prédiction")
            prediction_mode = st.radio(
                "Sélectionnez le mode",
                options=["simple", "progressive"],
                format_func=lambda x: " Standard" if x == "simple" else " Progressif",
                help="Standard = rapide | Progressif = plus prudent"
            )
            
            # Description du mode sélectionné
            if prediction_mode == "simple":
                st.success("**Rapide** : Toutes les prédictions en une fois ")
            else:
                st.warning(f"**Progressif** : Prédiction pas-à-pas ")

        # Bouton de génération centré et bien visible
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            generate_button = st.button(
                "Générer les prévisions",
                type="primary",
                use_container_width=True
            )

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════
        # ZONE DES RÉSULTATS
        # ═══════════════════════════════════════════════════════════════

        # État initial - Instructions
        if not generate_button:
            # Message principal
            st.info("👆 Configurez vos paramètres ci-dessus et cliquez sur **Générer les prévisions**")
            

        # Si le bouton est cliqué - GÉNÉRATION DES PRÉVISIONS
        else:
            # ═══════════════════════════════════════════════════════════
            # LA GÉNÉRATION des prévisions  
            # ═══════════════════════════════════════════════════════════
            if target_variable == 'Total Arrivées':
                if prediction_mode == "progressive":
                
                    try:
                        with st.spinner("⏳ Génération en cours..."):
                            start = time.time()
                            pred1 = predict_progressive(df, 'TES', horizon)
                            pred2= predict_progressive(df ,'MRE' , horizon)
                            predictions = (pred1.sum(axis=1) + pred2.sum(axis=1)).to_frame(name="somme_totale")
                            end = time.time()
                            temps_progressive = end - start
                        st.success("Prédictions générées avec succès !")
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des prévisions : {e}")
                        st.stop()

                else:
                # Mode simple - juste un spinner
                #st.subheader("⏳ Génération en cours...")
                    try :
                        with st.spinner("⏳ Génération en cours..."):
                            start_s = time.time()
                            pred1 = predict_simple(df, 'TES', 10)
                            pred2= predict_simple(df ,'MRE' , 10)
                            predictions = (pred1.sum(axis=1) + pred2.sum(axis=1)).to_frame(name="somme_totale")
                            end_s = time.time()
                            temps_simple = end_s - start_s

                        st.success("Prédictions générées avec succès !")
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des prévisions : {e}")
                        st.stop()
            else :
                # Afficher la progression pour mode progressif
                if prediction_mode == "progressive":
                    #st.subheader("⏳ Génération en cours...")
                    
                    try:
                        with st.spinner("⏳ Génération en cours..."):
                            start = time.time()
                            predictions = predict_progressive(df, target_variable, horizon)
                            end = time.time()
                            temps_progressive = end - start
                        st.success("Prédictions générées avec succès !")
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des prévisions : {e}")
                        st.stop()
                
                    
                else:
                    # Mode simple - juste un spinner
                    #st.subheader("⏳ Génération en cours...")
                    try :
                        with st.spinner("⏳ Génération en cours..."):
                            start_s = time.time()
                            predictions = predict_simple(df,target_variable, horizon)
                            end_s = time.time()
                            temps_simple = end_s - start_s

                        st.success("Prédictions générées avec succès !")
                    except Exception as e:
                        st.error(f"Erreur lors de la génération des prévisions : {e}")
                        st.stop()
            
            st.markdown("---")
            
            # ═══════════════════════════════════════════════════════════
            # GÉNÉRATION DES DONNÉES DE DÉMONSTRATION
            # ═══════════════════════════════════════════════════════════
            
            results_df = predictions 
            results_df.rename(columns={results_df.columns[0]: 'Prédiction'}, inplace=True)

            # ═══════════════════════════════════════════════════════════
            # AFFICHAGE DES RÉSULTATS - ONGLETS
            # ═══════════════════════════════════════════════════════════
            
            tab1, tab2, tab3 = st.tabs(["Visualisation", "Tableau détaillé", " Informations"])
            
            # ─────────────────────────────────────────────────────────
            # ONGLET 1 : VISUALISATION
            # ─────────────────────────────────────────────────────────
            with tab1:
                
                # Graphique principal
                fig = go.Figure()
                
            
                # Prédictions
                fig.add_trace(go.Scatter(
                    x=results_df.index,
                    y=results_df['Prédiction'].values,
                    mode='lines+markers',
                    name='Prédictions',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8)
                ))
                
                
                # Mise en forme
                fig.update_layout(
                    title=f"Prévisions - {target_variable.upper()} (Mode {prediction_mode})",
                    xaxis_title="Date",
                    yaxis_title="Valeur",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            
            # ─────────────────────────────────────────────────────────
            # ONGLET 2 : TABLEAU DÉTAILLÉ
            # ─────────────────────────────────────────────────────────
            with tab2:
                #st.subheader(" Détail des prédictions")
                # Statistiques du tableau
                st.markdown("### Statistiques")
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("Minimum", f"{min(results_df['Prédiction'].values):,.0f}")
                with stat_col2:
                    st.metric("Maximum", f"{max(results_df['Prédiction'].values):,.0f}")
                with stat_col3:
                    st.metric("Moyenne", f"{np.mean(results_df['Prédiction'].values):,.0f}")
                with stat_col4:
                    st.metric("Médiane", f"{np.median(results_df['Prédiction'].values):,.0f}")

                st.markdown("---")
                st.markdown(f"### Tables des predictions {target_variable}")
                # Formater le DataFrame pour l'affichage
                display_df = results_df.copy()
                display_df.reset_index(inplace=True)
                display_df.rename(columns={'index': 'Date','Prédiction':f'Prédiction_{target_variable}'}, inplace=True)
                #display_df['Date'] = display_df['Date'].dt.strftime('%B %Y')
                
                # Afficher le tableau
                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Prédiction": st.column_config.NumberColumn(
                            format="%.0f"
                        )
                    }
                )
                
                
                # Boutons d'export
                st.markdown("---")
                st.markdown("### Export des données")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name=f"predictions_{target_variable}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    excel_buffer = BytesIO()
                    # Écrire le DataFrame dans le buffer (pas 'csv')
                    display_df.to_excel(excel_buffer, index=False, engine='openpyxl')

                    # Revenir au début du buffer
                    excel_buffer.seek(0)

                    st.download_button(
                        label="Télécharger Excel",
                        data=excel_buffer,  # À remplacer par vraie conversion Excel
                        file_name=f"predictions_{target_variable}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
            # ─────────────────────────────────────────────────────────
            # ONGLET 3 : INFORMATIONS
            # ─────────────────────────────────────────────────────────
            with tab3:
                st.subheader(" Métadonnées de la prédiction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    ### Paramètres utilisés
                    """)
                    params_df = pd.DataFrame({
                        'Paramètre': [
                            'Variable cible',
                            'Mode de prédiction',
                            'Horizon',
                            'Date de début'
                        ],
                        'Valeur': [
                            target_variable,
                            'Standard' if prediction_mode == 'simple' else 'Progressif',
                            f'{horizon} mois',
                            start_date.strftime('%Y-%m-%d')
                        ]
                    })
                    st.dataframe(params_df, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    ###  Informations sur le modèle
                    """)
                    model_df = pd.DataFrame({
                        'Caractéristique': [
                            'Type de modèle',
                            'Ordre ARIMA',
                            'Ordre saisonnier',
                            'Saisonnalité',
                            'Temps d\'exécution',
                            'Date de génération'
                        ],
                        'Valeur': [
                            'SARIMA',
                            f"{(VARIABLES_CONFIG[target_variable]['order'] if target_variable !='Total Arrivées' else None)}",
                            f"{(VARIABLES_CONFIG[target_variable]['seasonal_order'] if target_variable !='Total Arrivées' else None )}",
                            '12 mois',
                            f"{(temps_progressive if prediction_mode=='progressive' else temps_simple):.3f} secondes",
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ]
                    })
                    st.dataframe(model_df, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                
                st.warning("""
                ### ⚠️ Notes importantes
                
                - **Fiabilité** : Ces prédictions sont basées sur les tendances historiques. L'incertitude augmente avec l'horizon.
                - **Mode progressif** : Chaque prédiction influence les suivantes, simulant un scénario sans données futures.
                - **Mise à jour recommandée** : Régénérez les prévisions mensuellement avec de nouvelles données pour maintenir la précision.
                - **Validation** : Comparez régulièrement les prédictions avec les réalisations pour évaluer la performance.
                """)
                

        
    
    # ==================== PAGE EXPLORATION AVIS ====================
    elif page == "Exploration les Avis":
        st.title("Exploration des Avis")
        st.markdown("---")
        
        # Fonction pour surligner les mots-clés
        def highlight_text(text, search_terms):
            """Surligne les termes recherchés dans le texte"""
            if not search_terms or pd.isna(text):
                return str(text) if pd.notna(text) else ""
            
            import re
            highlighted = str(text)
            
            # Diviser les termes de recherche par espaces
            terms = [term.strip() for term in search_terms.split() if term.strip()]
            
            # Surligner chaque terme (insensible à la casse)
            for term in terms:
                # Échapper les caractères spéciaux regex
                escaped_term = re.escape(term)
                # Créer un pattern insensible à la casse
                pattern = re.compile(f'({escaped_term})', re.IGNORECASE)
                # Remplacer avec le surlignage HTML
                highlighted = pattern.sub(
                    r'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: bold;">\1</mark>',
                    highlighted
                )
            
            return highlighted
        
        # Options de recherche
        st.subheader("Rechercher des avis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_text = st.text_input("Rechercher dans les avis (texte)", 
                                       help="Entrez des mots-clés séparés par des espaces. Ils seront surlignés dans les résultats.")
        
        with col2:
            # Vérifier si des notes sont disponibles
            if df_filtered['note'].notna().sum() > 0:
                min_note = st.slider("Note minimale", 1, 5, 1)
            else:
                min_note = None
                st.info("Pas de notes disponibles pour cette sélection")
        
        # Filtrer les avis
        if min_note is not None:
            df_search = df_filtered[df_filtered['note'] >= min_note].copy()
        else:
            df_search = df_filtered.copy()
        
        if search_text:
            df_search = df_search[
                df_search['text'].str.contains(search_text, case=False, na=False) |
                df_search['review'].str.contains(search_text, case=False, na=False)
            ]
        
        st.write(f"**{len(df_search)} avis trouvés**")
        
        # Afficher la légende si une recherche est active
        # if search_text and len(df_search) > 0:
        #     st.info(f"🔍 Les mots-clés recherchés sont surlignés en **jaune** dans les résultats ci-dessous.")
        
        # Affichage des avis
        for idx, row in df_search.head(50).iterrows():
            # Construire le titre de l'expander
            note_display = f"⭐ {row['note']}/5" if pd.notna(row['note']) else "⭐ N/A"
            expander_title = f"{note_display} - {row['city']} - {row['category']} - {row['date'].strftime('%d/%m/%Y')}"
            
            with st.expander(expander_title):
                sentiment_emoji = {
                    'positif': '',
                    'neutre': '',
                    'négatif': ''
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Sentiment:** {sentiment_emoji.get(row['sentiment'], '')} {row['sentiment'].capitalize()}")
                with col2:
                    st.write(f"**Plateforme:** {row['plateforme']}")
                with col3:
                    st.write(f"**Lieu:** {row.get('place_name', 'N/A')}")
                
                st.markdown("**Avis:**")
                
                # Récupérer le texte de l'avis
                avis_text = ""
                if pd.notna(row.get('review')):
                    avis_text = row['review']
                elif pd.notna(row.get('text')):
                    avis_text = row['text']
                
                if avis_text:
                    # Surligner les mots-clés si une recherche est active
                    if search_text:
                        highlighted_text = highlight_text(avis_text, search_text)
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                    else:
                        st.write(avis_text)
                else:
                    st.write("_Pas de texte disponible_")
        
        # Export des données filtrées
        st.markdown("---")
        st.subheader("Exporter les données")
        
        csv = df_search.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les avis filtrés (CSV)",
            data=csv,
            file_name=f'avis_filtres_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

else:
    st.error("⚠️ Impossible de charger les données. Veuillez vérifier que les fichiers 'reviews.csv' et 'arrivals.csv' sont présents dans le même répertoire que ce script.")
    st.info("📁 Fichiers requis:\n- reviews.csv (colonnes: city, category, place_name, date, note, review, plateforme, text, sentiment)\n- arrivals.csv (colonnes: date, TES, MRE, Total Arrivées, Nuité, Taux d'occupation)")
