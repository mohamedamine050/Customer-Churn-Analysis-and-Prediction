# Customer Churn Analysis and Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table des Matières

- [🎯 Objectif du Projet](#-objectif-du-projet)
- [🏆 Résultats Clés](#-résultats-clés)
- [📊 Dataset](#-dataset)
- [🚀 Installation et Configuration](#-installation-et-configuration)
- [📁 Structure du Projet](#-structure-du-projet)
- [🔧 Utilisation](#-utilisation)
- [🤖 Modèles Testés](#-modèles-testés)
- [📈 Performances](#-performances)
- [🌐 Application Streamlit](#-application-streamlit)
- [📊 Visualisations](#-visualisations)
- [🔮 Fonctionnalités](#-fonctionnalités)
- [🤝 Contribution](#-contribution)
- [📄 Licence](#-licence)

## 🎯 Objectif du Projet

Ce projet vise à **prédire le churn (désabonnement) des clients** d'une plateforme e-commerce en utilisant des techniques d'apprentissage automatique. L'objectif est d'identifier les clients à risque de churn pour permettre aux entreprises de prendre des mesures proactives de rétention.

### 🔍 Questions Métier Résolues

- **Qui sont les clients susceptibles de partir ?**
- **Quels facteurs influencent le plus le churn ?**
- **Comment prédire le comportement futur des clients ?**
- **Quelles stratégies de rétention adopter ?**

## 🏆 Résultats Clés

### 🎯 Meilleur Modèle : Random Forest Optimisé

| Métrique | Score | Description |
|----------|-------|-------------|
| **🎯 Précision** | **98.31%** | Exactitude globale des prédictions |
| **📈 ROC AUC** | **98.68%** | Capacité de discrimination |
| **🔄 Validation Croisée** | **97.86%** | Stabilité du modèle |
| **⚡ F1-Score (Churn)** | **95%** | Performance sur la classe minoritaire |

### 📊 Comparaison des Modèles

| Modèle | Précision | ROC AUC | CV Score |
|--------|-----------|---------|----------|
| **🏆 Random Forest** | **98.31%** | **98.68%** | **97.86%** |
| SVM | 91.65% | 93.93% | 90.70% |
| Gradient Boosting | 92.63% | 93.35% | 90.85% |
| Logistic Regression | 88.90% | 87.13% | 87.19% |

## 📊 Dataset

### 📈 Caractéristiques du Dataset

- **📋 Nombre total de clients** : 5,630
- **🔢 Variables prédictives** : 18 caractéristiques
- **📉 Taux de churn** : ~16.8%
- **📊 Type de données** : Mixte (numériques et catégorielles)

### 🔍 Variables Importantes

**Variables Numériques :**
- `Tenure` : Ancienneté du client
- `WarehouseToHome` : Distance entrepôt-domicile
- `HourSpendOnApp` : Temps passé sur l'application
- `OrderAmountHikeFromlastYear` : Augmentation du montant des commandes
- `OrderCount` : Nombre de commandes
- `DaySinceLastOrder` : Jours depuis la dernière commande

**Variables Catégorielles :**
- `PreferredLoginDevice` : Appareil de connexion préféré
- `PreferedOrderCat` : Catégorie de produits préférée
- `PreferredPaymentMode` : Mode de paiement préféré
- `Gender` : Genre
- `MaritalStatus` : Statut marital

## 🚀 Installation et Configuration

### 📋 Prérequis

```bash
Python 3.8+
pip (gestionnaire de packages Python)
```

### 🔧 Installation

1. **Cloner le repository :**
```bash
git clone https://github.com/Leangonplu/Ecommerce_Customer_Churn_Analysis_and_Prediction.git
cd Ecommerce_Customer_Churn_Analysis_and_Prediction
```

2. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

3. **Lancer l'application Streamlit :**
```bash
streamlit run streamlit_app.py
```

### 📦 Dépendances Principales

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
streamlit>=1.28.0
joblib>=1.3.0
openpyxl>=3.1.0
```

## 📁 Structure du Projet

```
📁 Ecommerce_Customer_Churn_Analysis_and_Prediction/
│
├── 📊 Data/
│   ├── E comm.xlsx                    # Dataset principal
│   └── E Commerce Dataset.xlsx        # Dataset alternatif
│
├── 📓 notebooks/
│   ├── 01-data-exploration.ipynb     # Exploration des données
│   ├── 02-data-preprocessing.ipynb   # Prétraitement
│   └── 03-model-training.ipynb       # Entraînement des modèles
│
├── 🤖 models/
│   ├── best_churn_model.pkl          # Modèle Random Forest optimisé
│   └── scaler.pkl                     # Normalisateur des données
│
├── 🎨 assets/
│   └── image.png                      # Images du projet
│
├── 🌐 streamlit_app.py                # Application web interactive
├── 📋 requirements.txt                # Dépendances Python
├── 📖 README.md                       # Documentation
└── 🏗️ src/                           # Code source (à développer)
```

## 🔧 Utilisation

### 1. 📊 Exploration des Données

```bash
jupyter notebook notebooks/01-data-exploration.ipynb
```

- Analyse descriptive du dataset
- Visualisations des distributions
- Identification des patterns

### 2. 🔄 Prétraitement

```bash
jupyter notebook notebooks/02-data-preprocessing.ipynb
```

- Nettoyage des données
- Gestion des valeurs manquantes
- Encodage des variables catégorielles
- Analyse de corrélation

### 3. 🤖 Entraînement des Modèles

```bash
jupyter notebook notebooks/03-model-training.ipynb
```

- Test de plusieurs algorithmes
- Optimisation des hyperparamètres
- Validation croisée
- Sauvegarde du meilleur modèle

### 4. 🌐 Application Interactive

```bash
streamlit run streamlit_app.py
```

- Interface utilisateur intuitive
- Prédictions en temps réel
- Visualisations interactives
- Analyse des performances

## 🤖 Modèles Testés

### 🧪 Algorithmes Évalués

1. **🌲 Random Forest** *(Sélectionné)*
   - Ensemble d'arbres de décision
   - Résistant au surapprentissage
   - Excellente performance

2. **🔍 Support Vector Machine (SVM)**
   - Classification par séparation optimale
   - Bonne performance générale

3. **⚡ Gradient Boosting**
   - Apprentissage séquentiel
   - Correction progressive des erreurs

4. **📈 Logistic Regression**
   - Modèle de référence (baseline)
   - Interprétabilité élevée

### 🎛️ Optimisation des Hyperparamètres

**Random Forest Optimisé :**
```python
{
    'n_estimators': 200,      # Nombre d'arbres
    'max_depth': 20,          # Profondeur maximale
    'min_samples_split': 2    # Échantillons min pour split
}
```

## 📈 Performances

### 🎯 Métriques Détaillées

**Classification Report (Random Forest) :**

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| **Pas de Churn** | 0.98 | 1.00 | 0.99 | 941 |
| **Churn** | 0.99 | 0.90 | 0.95 | 185 |
| **Moyenne** | 0.98 | 0.98 | 0.98 | 1126 |

### 📊 Matrice de Confusion

```
              Prédictions
Réel     │  Pas Churn │  Churn
─────────┼────────────┼────────
Pas Churn│     941    │    0
Churn    │      19    │  166
```

### 🔍 Analyse des Erreurs

- **Faux Positifs** : 0 (Excellent !)
- **Faux Négatifs** : 19 (Très faible)
- **Spécificité** : 100%
- **Sensibilité** : 89.7%

## 🌐 Application Streamlit

### 🖥️ Interface Utilisateur

L'application web propose 4 sections principales :

#### 🏠 Accueil
- Vue d'ensemble du projet
- Métriques du dataset
- Comparaison des modèles
- Performances du meilleur modèle

#### 📊 Exploration des Données
- Aperçu des données
- Statistiques descriptives
- Visualisations interactives
- Analyse de corrélation

#### 🤖 Prédictions
- Interface de saisie des caractéristiques client
- Prédiction en temps réel
- Probabilité de churn
- Recommandations personnalisées

#### 📈 Analyse des Résultats
- Courbe ROC
- Matrice de confusion
- Importance des variables
- Métriques de performance

### 🎮 Fonctionnalités Interactives

- **🎚️ Sliders** pour ajuster les paramètres
- **📈 Graphiques dynamiques** avec Plotly
- **🎯 Prédictions instantanées**
- **📊 Visualisations adaptatives**

## 📊 Visualisations

### 📈 Types de Graphiques Disponibles

1. **Distribution du Churn**
   - Graphique en secteurs
   - Répartition équilibrée

2. **Analyse Univariée**
   - Histogrammes des variables numériques
   - Graphiques en barres des variables catégorielles

3. **Analyse Bivariée**
   - Box plots par classe de churn
   - Matrices de corrélation

4. **Performance du Modèle**
   - Courbes ROC
   - Matrice de confusion
   - Importance des variables

## 🔮 Fonctionnalités

### ✅ Fonctionnalités Actuelles

- [x] Exploration complète des données
- [x] Prétraitement automatisé
- [x] Test de multiples algorithmes ML
- [x] Optimisation des hyperparamètres
- [x] Interface web interactive
- [x] Prédictions en temps réel
- [x] Visualisations dynamiques
- [x] Métriques de performance détaillées




**⭐ N'hésitez pas à mettre une étoile si ce projet vous a été utile ! ⭐**



</div>
