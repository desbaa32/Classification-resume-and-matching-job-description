"""
Module de prétraitement des données de CV et offres d'emploi

Ce module effectue:
1. Chargement des données brutes (CV et offres)
2. Nettoyage et prétraitement du texte
3. Normalisation et regroupement des catégories
4. Filtrage des catégories rares
5. Sauvegarde des données nettoyées
"""

import pandas as pd
import re
import emoji
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Configuration des chemins
DATA_DIR = Path('../data')

# Chemins pour les CV
RAW_RESUME_PATH = DATA_DIR / 'interim/combined_resume_final.csv'
PROCESSED_RESUME_PATH = DATA_DIR / 'processed/cleaned_combined_resume_final.csv'

# Chemins pour les offres
RAW_OFFER_PATH = DATA_DIR / 'interim/combined_offer_dataset.csv'
PROCESSED_OFFER_PATH = DATA_DIR / 'processed/cleaned_combined_offer_final.csv'


def load_data(file_path: Path) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV"""
    return pd.read_csv(file_path)


def read_data(data: pd.DataFrame, column_name: str) -> None:
    """Affichage des informations sur le dataset"""
    print("\n=== Aperçu des données ===")
    print(data.head(2))
    
    print("\n=== Informations sur les données ===")
    print(data.info())
    
    print(f"\n=== Taille du dataset: {data.shape} ===")
    
    print(f"\n=== Distribution des catégories ({column_name}) ===")
    print(data[column_name].value_counts().reset_index())


def normalize_categories(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Normalise les noms des catégories"""
    df = data.copy()
    
    # Normalisation de base
    df[column_name] = (
        df[column_name]
        .str.replace('_', ' ')
        .str.title()
        .str.strip()
    )
    
    # Gestion des variantes "Sr"
    df[column_name] = (
        df[column_name]
        .str.replace(r'\bSr\.?\b', 'Senior', regex=True)
        .str.replace(r'Senior\.', 'Senior', regex=True)
        .str.strip()
    )
    
    return df


def group_categories(
    data: pd.DataFrame, 
    category_column: str,
    category_mapping: Dict[str, str]
) -> pd.DataFrame:
    """Regroupe les catégories similaires selon le mapping prédéfini"""
    df = data.copy()
    df[category_column] = df[category_column].replace(category_mapping)
    return df


def filter_rare_categories(
    data: pd.DataFrame, 
    category_column: str,
    min_count: int = 30
) -> pd.DataFrame:
    """Filtre les catégories avec moins de min_count occurrences"""
    category_counts = data[category_column].value_counts()
    valid_categories = category_counts[category_counts >= min_count].index
    return data[data[category_column].isin(valid_categories)]


def remove_specific_categories(
    data: pd.DataFrame,
    category_column: str,
    categories_to_remove: List[str]
) -> pd.DataFrame:
    """Supprime des catégories spécifiques"""
    return data[~data[category_column].isin(categories_to_remove)]


def preprocess_text(text: str) -> str:
    """Nettoyage et prétraitement de base de texte pour l'analyse NLP"""
    if not isinstance(text, str):
        return ""
    
    # Mise en minuscules
    text = text.lower()
    
    # Suppression des bruits
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'<[^>]+>', '', text)  # Balises HTML
    text = re.sub(r'[\|@#$%^&*~_+=<>/\\{}¦©®™]', '', text)  # Caractères spéciaux
    text = re.sub(r'\b\d{6,}\b', ' ', text)  # Longues séquences numériques >=6
    text = re.sub(r'--+', ' ', text)  # Suites de tirets
    text = emoji.replace_emoji(text, replace='')  # Emojis
    text = re.sub(r'\s+', ' ', text).strip()  # Espaces multiples
    
    return text


def clean_dataset(data: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Nettoyage du dataset notamment la partie texte"""
    cleaned_data = data.copy()
    cleaned_data[text_column] = cleaned_data[text_column].apply(preprocess_text)
    return cleaned_data


def remove_duplicates(
    data: pd.DataFrame, 
    text_column: str,
    category_column: str
) -> pd.DataFrame:
    """Supprime les doublons exacts et textes dupliqués"""
    # Supprimer les doublons exacts
    data = data.drop_duplicates()
    
    # Supprimer les textes dupliqués (garder la première occurrence)
    data = data.drop_duplicates(subset=[text_column], keep='first')
    
    # Vérifier les textes dupliqués avec catégories différentes
    duplicates = data[data.duplicated(subset=[text_column], keep=False)]
    if not duplicates.empty:
        print(f"\nAttention: {len(duplicates)} textes dupliqués avec catégories différentes")
        print(duplicates[[category_column, text_column]].head(3))
    
    return data


def save_data(data: pd.DataFrame, file_path: Path) -> None:
    """Sauvegarde en csv du dataframe"""
    data.to_csv(file_path, index=False)


def process_resume_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Traite le dataset des CV"""
    print("\n" + "="*12 + "-"*6 + "="*12)
    print("TRAITEMENT DES CV")
    print("="*30)
    
    # Chargement des données
    raw_data = load_data(RAW_RESUME_PATH)
    print("\n=== Analyse des données brutes ===")
    read_data(raw_data, 'Category')
    
    # Définition du mapping de catégories
    category_mapping = {
        "Automation Testing": "Other IT",
        "Blockchain": "Other IT",
        "Director Of It": "Other IT",
        "Help Desk Analyst": "Other IT",
        "Help Desk Technician": "Other IT",
        "It Auditor": "Other IT",
        "Pmo": "Other IT",
        "Program Manager": "Other IT",
        "Technical Consultant": "Other IT",
        "Testing": "Other IT",
        "Desktop Support Technician": "Other IT",
        "It Support Specialist": "Other IT",
        "It Technician": "Other IT",
        "Project Coordinator": "Other IT",
        "It Director": "Other IT",
        "Scrum Master": "Project Manager",
        "Technical Project Manager": "Project Manager",
        "Civil Engineer": "Other No IT",
        "Electrical Engineering": "Other No IT",
        "Mechanical Engineer": "Other No IT",
        "Administrative Assistant": "Other No IT",
        "Arts": "Other No IT",
        "Customer Service Representative": "Other No IT",
        "Health And Fitness": "Other No IT",
        "Sales": "Other No IT",
        "Security Officer": "Other No IT",
        "Hr": "Other No IT",
        "Sales Associate": "Other No IT",
        "Advocate": "Other No IT",
        "Devops Engineer": "Big Data Cloud Developer",
        "Cloud Engineer": "Big Data Cloud Developer",
        "Hadoop": "Big Data Cloud Developer",
        "Hadoop Developer": "Big Data Cloud Developer",
        "Sr. Hadoop Developer": "Big Data Cloud Developer",
        "Etl Developer": "Big Data Cloud Developer",
        "It Business Analyst": "Data Analyst",
        "Data Science": "Data Scientist",
        "Consultant Consultant": "Consultant",
        "Consultant Consultant Consultant": "Consultant",
        "Contractor Contractor Contractor": "Consultant",
        "Independent Contractor": "Consultant",
        "Mobile App Developer (Ios/Android)": "Mobile Developer",
        "Android Application Developer": "Mobile Developer",
        "Android Developer": "Mobile Developer",
        "Business Analyst": "Data Analyst",
        "It Analyst": "Data Analyst",
        "Salesforce Administrator'": "Salesforce Developer",
        "Salesforce Admin/ Developer": "Salesforce Developer",
        "Salesforce Lightning Developer": "Salesforce Developer",
        "Frontend Developer": "Front End Developer",
        "Ui Developer Ui Developer Ui Developer": "Ui Developer",
        "Front- End Developer": "Front End Developer",
        "Front End Engineer": "Front End Developer",
        "Front End/Angular Developer": "Front End Developer",
        "Front End/Ui Developer": "Front End Ui Developer",
        "Front- End Web Developer": "Front End Developer",
        "Front-End Web Developer": "Front End Developer",
        "Developer Developer": "Software Developer",
        "Lead Front End Developer": "Lead Developer",
        "Lead Java Developer": "Lead Developer",
        "Freelance Web Developer": "Freelance Developer",
        "Freelance Front End Developer": "Freelance Developer",
        "It Security Engineer": "Security Engineer",
        "It Project Coordinator": "Project Manager",
        "Senior It Project Manager": "Senior Project Manager",
        "It Consultant": "Consultant",
        "It Specialist": "Other IT",
    }
    
    # Normalisation des catégories
    normalized_data = normalize_categories(raw_data, 'Category')
    print("\n=== Après normalisation des catégories ===")
    read_data(normalized_data, 'Category')

    # Regroupement des catégories similaires
    grouped_data = group_categories(normalized_data, 'Category', category_mapping)
    
    # Suppression de catégorie spécifique
    grouped_data = grouped_data[grouped_data['Category'] != "Web Developer,Software Developer"]
    
    print("\n=== Après regroupement des catégories ===")
    read_data(grouped_data, 'Category')

    # Filtrage des catégories rares
    filtered_data = filter_rare_categories(grouped_data, 'Category', min_count=30)
    print("\n=== Après filtrage des catégories rares ===")
    read_data(filtered_data, 'Category')
    
    # Suppression de catégories spécifiques supplémentaires
    categories_to_remove = ['It Security Analyst', 'It Project Manager']
    filtered_data = remove_specific_categories(
        filtered_data, 
        'Category', 
        categories_to_remove
    )
    
    # Nettoyage des données
    cleaned_data = clean_dataset(filtered_data, 'Resume')
    
    # Suppression des doublons
    cleaned_data = remove_duplicates(cleaned_data, 'Resume', 'Category')
    
    # Analyse finale
    print("\n=== Analyse des données nettoyées ===")
    read_data(cleaned_data, 'Category')
    
    # Sauvegarde
    save_data(cleaned_data, PROCESSED_RESUME_PATH)
    
    return raw_data, cleaned_data


def process_offer_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Traite le dataset des offres d'emploi"""
    print("\n" + "="*12 + "-"*6 + "="*12)
    print("TRAITEMENT DES OFFRES D'EMPLOI")
    print("="*30)
    
    # Chargement des données
    raw_data = load_data(RAW_OFFER_PATH)
    print("\n=== Analyse des données brutes ===")
    read_data(raw_data, 'Job Title_Category')
    
    # Normalisation des catégories
    normalized_data = normalize_categories(raw_data, 'Job Title_Category')
    print("\n=== Après normalisation des catégories ===")
    read_data(normalized_data, 'Job Title_Category')

    # Suppression des doublons
    dedup_data = remove_duplicates(
        normalized_data, 
        'Job Description', 
        'Job Title_Category'
    )
    
    # Nettoyage des données
    cleaned_data = clean_dataset(dedup_data, 'Job Description')
    
    # Analyse finale
    print("\n=== Analyse des données nettoyées ===")
    read_data(cleaned_data, 'Job Title_Category')
    
    # Sauvegarde
    save_data(cleaned_data, PROCESSED_OFFER_PATH)
    
    return raw_data, cleaned_data


def main() -> None:
    """
    Flux principal de prétraitement des données
    Traite à la fois les CV et les offres d'emploi
    """
    # Traitement des CV
    resume_raw, resume_clean = process_resume_data()
    
    # Traitement des offres
    offer_raw, offer_clean = process_offer_data()
    
    print("\n" + "="*20 + "-"*10 + "="*20)
    print("PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS")
    print("="*50)
    print(f"CV nettoyés sauvegardés: {PROCESSED_RESUME_PATH}")
    print(f"Offres nettoyées sauvegardées: {PROCESSED_OFFER_PATH}")


if __name__ == "__main__":
    main()