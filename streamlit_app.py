import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Churn E-commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🛍️ Analyse et Prédiction de Churn Client E-commerce")
st.markdown("---")

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charger et prétraiter les données"""
    try:
        df = pd.read_excel('Data/E comm.xlsx')
        
        # Prétraitement
        df.drop(columns='CustomerID', inplace=True, errors='ignore')
        
        # Remplir les valeurs manquantes
        numeric_columns = ['HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'Tenure', 
                          'WarehouseToHome', 'DaySinceLastOrder', 'OrderCount', 'CouponUsed']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Encoder les variables catégorielles
        from sklearn import preprocessing
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'Churn':
                label_encoder = preprocessing.LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col].astype(str))
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    """Charger le modèle et le scaler"""
    try:
        model = joblib.load('models/best_churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Modèle non trouvé. Veuillez d'abord entraîner le modèle: {e}")
        return None, None

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.selectbox(
    "Choisir une page",
    ["🏠 Accueil", "📊 Exploration des Données", "🤖 Prédictions", "📈 Analyse des Résultats"]
)

# Charger les données
df = load_data()

if df is not None:
    
    # PAGE ACCUEIL
    if page == "🏠 Accueil":
        st.header("Bienvenue dans l'Analyse de Churn Client")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Total Clients",
                value=f"{len(df):,}",
                delta="Dataset complet"
            )
        
        with col2:
            churn_rate = (df['Churn'].sum() / len(df)) * 100 if 'Churn' in df.columns else 0
            st.metric(
                label="📉 Taux de Churn",
                value=f"{churn_rate:.1f}%",
                delta=f"{df['Churn'].sum()} clients" if 'Churn' in df.columns else "N/A"
            )
        
        with col3:
            features_count = len(df.columns) - 1 if 'Churn' in df.columns else len(df.columns)
            st.metric(
                label="🎯 Variables",
                value=features_count,
                delta="Caractéristiques"
            )
        
        st.markdown("---")
        
        # Description du projet
        st.subheader("📝 À propos du projet")
        st.markdown("""
        Cette application analyse le comportement des clients e-commerce pour prédire le churn (abandon).
        
        **Fonctionnalités disponibles:**
        - 📊 **Exploration des données**: Analyse descriptive et visualisations
        - 🤖 **Prédictions**: Prédire le churn pour de nouveaux clients
        - 📈 **Analyse des résultats**: Évaluation des performances du modèle
        """)
        
        # Section du meilleur modèle
        st.markdown("---")
        st.subheader("🏆 Meilleur Modèle Sélectionné")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🌲 Random Forest Optimisé
            
            Après avoir testé plusieurs algorithmes d'apprentissage automatique, le **Random Forest** 
            a été identifié comme le modèle le plus performant pour prédire le churn client.
            
            **Pourquoi Random Forest ?**
            - ✅ Excellente performance sur les données tabulaires
            - ✅ Résistant au surapprentissage  
            - ✅ Gère bien les variables catégorielles et numériques
            - ✅ Fournit l'importance des variables
            """)
        
        with col2:
            # Métriques du meilleur modèle
            st.success("**🎯 PERFORMANCES**")
            
            st.metric("🎯 Précision", "98.31%", "+0.86%")
            st.metric("📈 ROC AUC", "98.68%", "+1.55%")
            st.metric("🔄 CV Score", "97.86%", "Excellent")
        
        # Comparaison des modèles
        st.markdown("---")
        st.subheader("📊 Comparaison des Modèles Testés")
        
        # Données de comparaison des modèles
        model_comparison = {
            'Modèle': ['Random Forest', 'SVM', 'Gradient Boosting', 'Logistic Regression'],
            'Précision (%)': [98.31, 91.65, 92.63, 88.90],
            'ROC AUC (%)': [98.68, 93.93, 93.35, 87.13],
            'CV Score (%)': [97.86, 90.70, 90.85, 87.19]
        }
        
        comparison_df = pd.DataFrame(model_comparison)
        
        # Affichage du tableau
        st.dataframe(comparison_df, use_container_width=True)
        
        # Graphique comparatif
        fig = px.bar(
            comparison_df.melt(id_vars='Modèle', var_name='Métrique', value_name='Score'),
            x='Modèle',
            y='Score',
            color='Métrique',
            title="Comparaison des Performances des Modèles",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            xaxis_title="Modèles",
            yaxis_title="Score (%)",
            legend_title="Métriques",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique de distribution du churn
        if 'Churn' in df.columns:
            st.subheader("📊 Distribution du Churn")
            
            fig = px.pie(
                values=df['Churn'].value_counts().values,
                names=['Pas de Churn', 'Churn'],
                title="Répartition des Clients",
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE EXPLORATION DES DONNÉES
    elif page == "📊 Exploration des Données":
        st.header("📊 Exploration des Données")
        
        # Affichage des données
        st.subheader("🔍 Aperçu des Données")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Informations générales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Informations Générales")
            st.write(f"**Nombre de lignes**: {df.shape[0]:,}")
            st.write(f"**Nombre de colonnes**: {df.shape[1]}")
            st.write(f"**Valeurs manquantes**: {df.isnull().sum().sum()}")
        
        with col2:
            st.subheader("📊 Types de Données")
            type_counts = df.dtypes.value_counts()
            fig = px.bar(
                x=type_counts.index.astype(str),
                y=type_counts.values,
                title="Distribution des Types de Données"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques descriptives
        st.subheader("📈 Statistiques Descriptives")
        numeric_df = df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe(), use_container_width=True)
        
        # Matrice de corrélation
        if len(numeric_df.columns) > 1:
            st.subheader("🔗 Matrice de Corrélation")
            
            # Calculer la corrélation
            corr_matrix = numeric_df.corr()
            
            # Créer le heatmap avec Plotly
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Variables", y="Variables", color="Corrélation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            fig.update_layout(title="Matrice de Corrélation des Variables Numériques")
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des variables numériques
        st.subheader("📊 Distribution des Variables")
        
        if len(numeric_df.columns) > 0:
            selected_var = st.selectbox("Choisir une variable à analyser:", numeric_df.columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogramme
                fig1 = px.histogram(
                    df, 
                    x=selected_var, 
                    title=f"Distribution de {selected_var}",
                    nbins=30
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Box plot si Churn existe
                if 'Churn' in df.columns:
                    fig2 = px.box(
                        df, 
                        x='Churn', 
                        y=selected_var,
                        title=f"{selected_var} par Statut de Churn"
                )
                fig2.update_xaxes(tickvals=[0, 1], ticktext=['Pas de Churn', 'Churn'])
                st.plotly_chart(fig2, use_container_width=True)    # PAGE PRÉDICTIONS
    elif page == "🤖 Prédictions":
        st.header("🤖 Prédictions de Churn")
        
        # Charger le modèle
        model, scaler = load_model()
        
        if model is not None and scaler is not None:
            st.success("✅ Modèle chargé avec succès!")
            
            st.subheader("📋 Entrer les Données Client")
            
            # Formulaire de prédiction
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                # Récupérer les colonnes numériques (excluant Churn)
                feature_columns = [col for col in df.columns if col != 'Churn']
                
                inputs = {}
                
                # Créer les inputs pour chaque caractéristique
                for i, col in enumerate(feature_columns):
                    column_to_use = col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3)
                    
                    with column_to_use:
                        if df[col].dtype in ['int64', 'float64']:
                            min_val = float(df[col].min())
                            max_val = float(df[col].max())
                            mean_val = float(df[col].mean())
                            
                            inputs[col] = st.number_input(
                                f"{col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                help=f"Valeur entre {min_val:.2f} et {max_val:.2f}"
                            )
                        else:
                            unique_vals = df[col].unique()
                            inputs[col] = st.selectbox(f"{col}", unique_vals)
                
                submitted = st.form_submit_button("🔮 Prédire le Churn")
                
                if submitted:
                    # Préparer les données pour la prédiction
                    input_data = np.array([[inputs[col] for col in feature_columns]])
                    
                    # Normaliser les données
                    input_scaled = scaler.transform(input_data)
                    
                    # Faire la prédiction
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0]
                    
                    # Afficher les résultats
                    st.markdown("---")
                    st.subheader("🎯 Résultat de la Prédiction")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error("⚠️ **RISQUE DE CHURN ÉLEVÉ**")
                            st.markdown(f"**Probabilité de churn**: {probability[1]:.1%}")
                        else:
                            st.success("✅ **CLIENT FIDÈLE**")
                            st.markdown(f"**Probabilité de rétention**: {probability[0]:.1%}")
                    
                    with col2:
                        # Graphique de probabilité
                        prob_df = pd.DataFrame({
                            'Statut': ['Pas de Churn', 'Churn'],
                            'Probabilité': probability
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x='Statut',
                            y='Probabilité',
                            title="Probabilités de Prédiction",
                            color='Statut',
                            color_discrete_map={'Pas de Churn': '#00CC96', 'Churn': '#EF553B'}
                        )
                        fig.update_yaxes(tickformat='.1%')
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Modèle non disponible. Veuillez d'abord entraîner le modèle dans le notebook 03-model-training.ipynb")
    
    # PAGE ANALYSE DES RÉSULTATS
    elif page == "📈 Analyse des Résultats":
        st.header("📈 Analyse des Résultats du Modèle")
        
        model, scaler = load_model()
        
        if model is not None and scaler is not None:
            st.success("✅ Modèle disponible pour l'analyse!")
            
            # Informations sur le modèle
            st.subheader("🤖 Informations sur le Modèle")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Type de modèle**: {type(model).__name__}")
                st.info(f"**Nombre de features**: {len([col for col in df.columns if col != 'Churn'])}")
            
            with col2:
                if hasattr(model, 'n_estimators'):
                    st.info(f"**Nombre d'estimateurs**: {model.n_estimators}")
                if hasattr(model, 'max_depth'):
                    st.info(f"**Profondeur max**: {model.max_depth}")
            
            # Importance des features (si disponible)
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Importance des Variables")
                
                feature_names = [col for col in df.columns if col != 'Churn']
                importances = model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'Variable': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(10),
                    x='Importance',
                    y='Variable',
                    orientation='h',
                    title="Top 10 des Variables les Plus Importantes"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Conseils d'action
            st.subheader("💡 Recommandations d'Action")
            
            recommendations = [
                "🎯 **Cibler les clients à risque**: Utilisez le modèle pour identifier les clients susceptibles de partir",
                "📞 **Campagnes de rétention**: Contactez proactivement les clients à haut risque",
                "🎁 **Offres personnalisées**: Proposez des remises ou avantages aux clients identifiés",
                "📊 **Surveillance continue**: Réévaluez régulièrement le risque de churn",
                "🔄 **Amélioration du service**: Concentrez-vous sur les facteurs les plus importants"
            ]
            
            for rec in recommendations:
                st.markdown(rec)
        
        else:
            st.warning("⚠️ Aucun modèle disponible pour l'analyse. Entraînez d'abord un modèle.")

else:
    st.error("❌ Impossible de charger les données. Vérifiez que le fichier 'Data/E comm.xlsx' existe.")

# Footer
st.markdown("---")
st.markdown("📊 **Analyse de Churn E-commerce** - Créé avec Streamlit")