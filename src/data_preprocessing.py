"""
Module de prétraitement des données de CV pour classification

Ce module effectue:
1. Chargement des données brutes
2. Nettoyage et prétraitement du texte
3. Analyse des données
4. Sauvegarde des données nettoyées
"""

import pandas as pd
import re
import emoji
from pathlib import Path
from typing import Dict, Tuple

# Configuration des chemins
DATA_DIR = Path('../data')
RAW_PATH = DATA_DIR / 'interim/combined_resume_final.csv'
PROCESSED_PATH = DATA_DIR / 'processed/cleaned_combined_resume_final.csv'


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV
    """
    return pd.read_csv(file_path)


def read_data(data: pd.DataFrame) -> None:
    """
    Affichage des informations sur le dataset
    
    """
    print("\n _ Aperçu des données _")
    print(data.head(2))
    
    print("\n_ Informations sur les données _")
    print(data.info())
    
    print("\n_ Taille du dataset _")
    print(data.shape)
    
    print("\n_ Distribution des catégories _")
    print(data['Category'].value_counts().reset_index())

def normalize_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms des catégories"""
    df = data.copy()
    
    # Normalisation de base
    df['Category'] = (
        df['Category']
        .str.replace('_', ' ')
        .str.title()
        .str.strip()
    )
    
    # Gestion des variantes "Sr"
    df['Category'] = (
        df['Category']
        .str.replace(r'\bSr\.?\b', 'Senior', regex=True)
        .str.replace(r'Senior\.', 'Senior', regex=True)
        .str.strip()
    )
    
    return df


def group_categories(data: pd.DataFrame,category_mapping: Dict[str, str]) -> pd.DataFrame:
    """Regroupe les catégories  similaire  selon le mapping prédéfini"""
    df = data.copy()
    df['Category'] = df['Category'].replace(category_mapping)
    return  df['Category']


def filter_rare_categories(data: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """Filtre les catégories avec moins de min_count occurrences"""
    category_counts = data['Category'].value_counts()
    valid_categories = category_counts[category_counts >= min_count].index
    return data[data['Category'].isin(valid_categories)]

def preprocess_text(text: str) -> str:
    """
    Nettoyage et prétraitement de base de texte pour l'analyse NLP
    
    
    """
    if not isinstance(text, str):
        return ""
    
    # Mise en minuscules
    text = text.lower()
    
    # Suppression 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'<[^>]+>', '', text)  # Balises HTML
    text = re.sub(r'[\|@#$%^&*~_+=<>/\\{}¦©®™]', '', text)  # Caractères spéciaux
    text = re.sub(r'\b\d{6,}\b', ' ', text)  # Longues séquences numériques >=6
    text = re.sub(r'--+', ' ', text)  # Suites de tirets
    text = emoji.replace_emoji(text, replace='') #  emojis
    text = re.sub(r'\s+', ' ', text).strip()  # des espaces multiples
    
    return text


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage du dataset nottament la partie resume
    
    """
    cleaned_data = data.copy()
    
    # Prétraitement du texte
    cleaned_data['Resume'] = cleaned_data['Resume'].apply(preprocess_text)
    
    return cleaned_data


def save_data(data: pd.DataFrame, file_path: Path) -> None:
    """
    Sauvegarde en csv du dataframe
    
    """
    data.to_csv(file_path, index=False)


def main() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flux principal Normalisation  de prétraitement des données
    Ce module effectue:
    1. Chargement des données brutes
    2. Nettoyage et prétraitement du texte
    3. Normalisation et regroupement des catégories
    4. Filtrage des catégories rares
    5. Pretraitement  des données
    6. Sauvegarde des données nettoyées
    Returns:
        Tuple contenant (données brutes, données nettoyées)
    """
    # Chargement des données
    raw_data = load_data(RAW_PATH)
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
    },
    # Analyse des données brutes
    print("\n=== Analyse des données brutes ===")
    read_data(raw_data)
     # Normalisation des catégories
    normalized_data = normalize_categories(raw_data)
    print("\n=== Après normalisation des catégories ===")
    read_data(normalized_data)

    # Regroupement des catégories similaire
    grouped_data = group_categories(normalized_data,category_mapping) # type: ignore
    print("\n=== Après regroupement des catégories ===")
    read_data(grouped_data)

    # Filtrage des catégories rares restant 
    filtered_data = filter_rare_categories(grouped_data)
    print("\n=== Après filtrage des catégories rares ===")
    read_data(filtered_data)
    # Nettoyage des données
    cleaned_data = clean_dataset(filtered_data)
    
    # Analyse des données nettoyées
    print("\n=== Analyse des données nettoyées ===")
    read_data(cleaned_data)
    
    # Sauvegarde
    save_data(cleaned_data, PROCESSED_PATH)
    
    return raw_data, cleaned_data


if __name__ == "__main__":
    raw_df, cleaned_df = main()