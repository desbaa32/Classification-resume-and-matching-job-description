# Classification-resume-and-matching-job-description
#### Modélisation d’un système automatisé de classification et d’appariement CV-offres basées sur les profils:Projet Memoire premiere annee de master

## Description

Système intelligent de matching entre CV et offres d'emploi utilisant le traitement du langage naturel (NLP) et l'apprentissage profond. Ce projet combine des techniques avancées de NLP avec des modèles de deep learning pour classer et apparier automatiquement des CV avec des offres d'emploi pertinentes.

## Fonctionnalités

- **Classification automatique** des CV et offres d'emploi
- **Matching sémantique** basé sur les embeddings de texte
- **Architecture hybride** combinant classification et similarité sémantiqueng

## Structure du Projet

```
cv-job-matching/
├── data/
│   ├── raw/                 # Données brutes
│   ├── interim/            # Données intermédiaires
│   └── processed/          # Données nettoyées
├── notebooks/
|   ├── Google Colab        #fichier nootebooks executer avec Google colab
|       ├── classification_resume.ipynb    # Classification des CV nootebooks
│       ├── classification_offer.ipynb    # Classification des offres notebooks
│       └── matching_model.ipynb          # Système de matching notebooks
│   ├── data_preprocessing.ipynb 
|   ├── dataset_consolidation.ipynb
|   └──visualization_daataExploraion
|

├── src/
|   ├── models/
│       ├── classification_resume.py    # Classification des CV
│       ├── classification_offer.py     # Classification des offres
│       └── matching_model.py           # Système de matching
|   ├── data_preprocessing.py       # Pretraitement data
│   ├── classification_run.py       # Script d'exécution classification
│   └── run_matching.py             # Script d'exécution matching
├── requirements.txt
└── README.md
```

## Installation

1. Cloner le repository
```bash
git clone <repository-url>
cd Classification-resume-and-matching-job-description
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Préparer les données
- Placer les fichiers de données dans le dossier `data/processed/`
- Les fichiers doivent s'appeler :
  - `cleaned_combined_resume_final.csv` pour les CV
  - `cleaned_combined_offer_final.csv` pour les offres

## Utilisation

### Classification

Pour exécuter les modèles de classification :

```bash
# Classification des CV seulement
python src/classification_run.py --resume

# Classification des offres seulement
python src/classification_run.py --offer

```

### Matching

Pour utiliser le système de matching :

```bash
# Mode interactif
python src/run_matching.py

# Mode évaluation
python src/run_matching.py --eval

# Matching spécifique
python src/run_matching.py --text "Votre texte ici" --type cv --top_k 5
```

## API des Modules

### classification_resume.py
Fonction principale : `main()`
- Charge et prétraite les données CV
- Entraîne un modèle de classification
- Évalue et sauvegarde le modèle

### classification_offer.py  
Fonction principale : `main()`
- Charge et prétraite les données offres
- Entraîne un modèle de classification
- Évalue et sauvegarde le modèle

### matching_model.py
Classe principale : `HybridMatcher`
Méthodes principales :
- `hybrid_matching(query_text, query_type, top_k=5)`: Matching hybride
- `evaluate_hybrid_system(test_cases)`: Évaluation du système
- `display_evaluation_results(results)`: Affichage des résultats

## Performance

- Précision classification CV: ~85%
- Précision classification offres: ~82%
- Matching sémantique: résultats "passables" (en cours d'amélioration)

## Améliorations Futures

- Fine-tuning des modèles de embedding
- Intégration de connaissances métier
- Interface utilisateur web
- API RESTful

## Auteur

Projet développé dans le cadre d'un mémoire de Master en Informatique par @desbaa32

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.