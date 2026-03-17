# Hackathon_Explainability_AI_Groupe13

README : Système d'Aide à la Rétention des Talents (RH-AI)
1. Objectifs du projet
L'objectif de cette solution est d'aider les départements des Ressources Humaines à anticiper le turnover au sein de l'entreprise. En combinant l'analyse prédictive (Machine Learning) et l'IA générative (LLM), le système permet de :

Identifier proactivement les profils à risque de démission via un score de probabilité.

Comprendre les facteurs de risque spécifiques à chaque employé (ex: charge de travail, manque d'engagement, ancienneté).

Proposer des pistes de rétention personnalisées grâce à un assistant IA expert.

2. Périmètre (Scope)
Data Processing : Nettoyage, normalisation et feature engineering des données RH brutes.

Modélisation : Classification supervisée (RandomForest) pour l'attribution d'un score de risque de départ.

Interface LLM : Utilisation de l'API Groq pour générer des recommandations RH basées sur les données traitées.

Limites : Le système est un outil d'aide à la décision. Il ne remplace pas le dialogue humain et doit être utilisé dans le respect de la confidentialité des données et de l'éthique RH.

3. Persona : Qui utilise cet outil ?
Responsable RH / Manager : Utilisateur principal qui consulte les listes d'employés à risque, examine les détails d'un profil spécifique et s'appuie sur les conseils générés par l'IA pour préparer ses entretiens de rétention.

4. Instructions d'utilisation
Prérequis
Python 3.x

Dépendances : pandas, scikit-learn, numpy, requests

Clé API : Une clé valide pour l'API Groq.

Installation
Clonez ce dépôt.

Installez les dépendances :

pip install -r requirements.txt


Flux d'exécution :
Prétraitement : Lancez le script de pipeline pour transformer le dataset HRDataset_v14.csv et entraîner le modèle.

Inférence : Utilisez la fonction de prédiction pour obtenir le score de risque d'un employé via son EmpID.

Recommandation : Appelez le module LLM avec l'ID employé pour recevoir l'analyse détaillée et les conseils stratégiques :


Points d'attention (Éthique & Sécurité)
Biais : Ce modèle a été entraîné sur des données historiques. Veillez à ne pas automatiser des décisions basées uniquement sur ces résultats sans examen humain.

Sécurité : Ne jamais inclure votre clé API dans le code source ou dans un commit Git.


Shéma d'architecture du système :

```mermaid
graph TD
    %% Couche Données
    subgraph Data_Layer [Couche Données]
        A[Dataset RH HRDataset_v14.csv] --> B[Pipeline de Prétraitement]
        B --> C[Imputation, Scaling, OneHotEncoding]
    end

    %% Couche Modèle
    subgraph ML_Layer [Couche Modèle]
        C --> D[RandomForest Classifier]
        D -->|Génère| E[Score de Risque]
        D -->|Analyse| F[Importances des Variables]
    end

    %% Couche LLM (Bloc 4 détaillé)
    subgraph LLM_Layer [Couche d'Explicabilité & LLM]
        E --> G[Prompt Engineering]
        F --> G
        G --> H{API Groq LLM}
        H -->|Instruction système: Expert RH| I[Recommandations Personnalisées]
    end

    %% Output
    I --> J[Dashboard RH: Actions à entreprendre]

    %% Styles
    style H fill:#f96,stroke:#333,stroke-width:2px
    style D fill:#6cf,stroke:#333,stroke-width:2px
