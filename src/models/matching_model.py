"""
Système de matching hybride CV-offres avec classification
"""

"""
Module de matching hybride CV-offres
"""

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer
import faiss
from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import logging
from typing import List, Dict, Tuple, Optional, Any

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridMatcher:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialise le système de matching hybride
        """
        # Configuration par défaut
        self.default_config = {
            'data_paths': {
                'resume': 'data/processed/cleaned_combined_resume_final.csv',
                'offer': 'data/processed/cleaned_combined_offer_final.csv',
                'model_dir': 'models/saved_model',
                'matching_dir': 'models/matching_data'
            },
            'sampling_params': {
                'total_cv_samples': 1200,
                'total_offer_samples': 800,
                'min_threshold': 35,
                'rare_threshold_cv': 40,
                'rare_threshold_offer': 50
            },
            'matching_params': {
                'top_k': 5,
                'confidence_threshold': 0.7,
                'max_length': 256,
                'batch_size': 32
            }
        }
        
        # Appliquer la configuration personnalisée
        self.config = {**self.default_config, **(config or {})}
        
        # Initialiser les attributs
        self.cv_dataset = None
        self.offer_dataset = None
        self.cv_index = None
        self.offer_index = None
        self.cv_embeddings = None
        self.offer_embeddings = None
        self.cv_id_to_index = None
        self.offer_id_to_index = None
        self.sbert_model = None
        self.cv_classifier = None
        self.offer_classifier = None
        self.cv_le = None
        self.offer_le = None
        self.tokenizer = None
        self.distilbert_model = None
        
    def generate_id(self, text: str) -> str:
        """
        Génère un ID unique à partir du texte
        
        Args:
            text: Texte à hasher
            
        Returns:
            ID unique (12 premiers caractères du hash MD5)
        """
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def balanced_sampling(self, dataset: pd.DataFrame, total_samples: int, 
                         threshold_min: int, threshold_rare: int, 
                         category_column: str = 'Category') -> pd.DataFrame:
        """
        Génère un échantillon équilibré selon les seuils spécifiés
        """
        # Calculer la fréquence des catégories
        category_counts = dataset[category_column].value_counts().reset_index()
        category_counts.columns = [category_column, 'Count']
        
        # Initialiser un DataFrame pour l'échantillon final
        sampled_data = pd.DataFrame()
        
        # Étape 1 : Prendre TOUTES les catégories rares
        rare_categories = category_counts[category_counts['Count'] < threshold_rare][category_column]
        for category in rare_categories:
            category_data = dataset[dataset[category_column] == category]
            sampled_data = pd.concat([sampled_data, category_data])
        
        # Étape 2 : Échantillonner les catégories fréquentes
        frequent_categories = category_counts[category_counts['Count'] >= threshold_rare][category_column]
        remaining_samples = total_samples - len(sampled_data)
        
        for category in frequent_categories:
            category_data = dataset[dataset[category_column] == category]
            
            # Calculer la proportion cible basée sur la fréquence relative
            proportion = len(category_data) / len(dataset)
            target_samples = max(int(proportion * remaining_samples), threshold_min)
            
            # Prendre le minimum entre le target et le disponible
            n_samples = min(target_samples, len(category_data))
            
            # Échantillonner
            sampled_category = category_data.sample(n=n_samples, random_state=42)
            sampled_data = pd.concat([sampled_data, sampled_category])
        
        # Étape 3 : Ajuster si le total dépasse total_samples
        if len(sampled_data) > total_samples:
            excess = len(sampled_data) - total_samples
            large_categories = sampled_data[category_column].value_counts().index.tolist()
            to_remove = pd.DataFrame()
            
            for cat in large_categories:
                if excess <= 0:
                    break
                    
                cat_data = sampled_data[sampled_data[category_column] == cat]
                # Ne pas descendre en dessous du threshold_min pour les catégories fréquentes
                min_threshold = threshold_min if cat in frequent_categories.tolist() else 0
                n_remove = min(excess, len(cat_data) - min_threshold)
                
                if n_remove > 0:
                    removed = cat_data.sample(n=n_remove, random_state=42)
                    to_remove = pd.concat([to_remove, removed])
                    excess -= n_remove
            
            sampled_data = sampled_data.drop(to_remove.index)
        
        return sampled_data.reset_index(drop=True)
    
    def load_data_sampling(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Charge les données originales et applique l'échantillonnage équilibré
        
        Returns:
            Tuple avec (cv_dataset, offer_dataset, cv_id_to_index, offer_id_to_index)
        """
        # Charger les datasets originaux
        resume_dataset = pd.read_csv(self.config['data_paths']['resume'])
        offer_dataset = pd.read_csv(self.config['data_paths']['offer'])
        
        # Paramètres d'échantillonnage
        sampling_params = self.config['sampling_params']
        
        # Créer des échantillons équilibrés
        cv_dataset = self.balanced_sampling(
            resume_dataset, 
            sampling_params['total_cv_samples'], 
            sampling_params['min_threshold'], 
            sampling_params['rare_threshold_cv']
        )
        
        offer_dataset = self.balanced_sampling(
            offer_dataset, 
            sampling_params['total_offer_samples'], 
            sampling_params['min_threshold'], 
            sampling_params['rare_threshold_offer'],
            'Job Title_Category'
        )
        
        # Nettoyage supplémentaire
        cv_dataset = cv_dataset.dropna(subset=['Resume']).reset_index(drop=True)
        offer_dataset = offer_dataset.dropna(subset=['Job Description']).reset_index(drop=True)
        
        # Création d'IDs persistants basés sur le contenu
        cv_dataset['cv_id'] = cv_dataset['Resume'].apply(self.generate_id)
        offer_dataset['offer_id'] = offer_dataset['Job Description'].apply(self.generate_id)
        
        # Création de mappings pour retrouver les index à partir des IDs
        cv_id_to_index = {id_: idx for idx, id_ in enumerate(cv_dataset['cv_id'])}
        offer_id_to_index = {id_: idx for idx, id_ in enumerate(offer_dataset['offer_id'])}
        
        logger.info(f"Échantillon CVs créé: {len(cv_dataset)} éléments")
        logger.info(f"Échantillon offres créé: {len(offer_dataset)} éléments")
        
        return cv_dataset, offer_dataset, cv_id_to_index, offer_id_to_index
    
    def generate_embeddings(self, cv_dataset: pd.DataFrame, offer_dataset: pd.DataFrame) -> Tuple:
        """
        Génère les embeddings et les index FAISS pour les datasets fournis
        """
        # Charger le modèle SBERT
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Génération des embeddings pour tous les CV
        logger.info("Génération des embeddings pour les CV...")
        cv_texts = cv_dataset['Resume'].tolist()
        cv_embeddings = self.sbert_model.encode(cv_texts, convert_to_tensor=True, show_progress_bar=True)
        
        # Génération des embeddings pour toutes les offres
        logger.info("Génération des embeddings pour les offres...")
        offer_texts = offer_dataset['Job Description'].tolist()
        offer_embeddings = self.sbert_model.encode(offer_texts, convert_to_tensor=True, show_progress_bar=True)
        
        # Conversion en numpy array pour FAISS
        cv_embeddings_np = cv_embeddings.cpu().numpy()
        offer_embeddings_np = offer_embeddings.cpu().numpy()
        
        # Normalisation des vecteurs pour la similarité cosinus
        faiss.normalize_L2(cv_embeddings_np)
        faiss.normalize_L2(offer_embeddings_np)
        
        # Création des index FAISS pour une recherche rapide
        dimension = cv_embeddings_np.shape[1]
        cv_index = faiss.IndexFlatIP(dimension)
        offer_index = faiss.IndexFlatIP(dimension)
        
        # Ajout des vecteurs aux index
        cv_index.add(cv_embeddings_np)
        offer_index.add(offer_embeddings_np)
        
        logger.info("Embeddings et index FAISS générés avec succès")
        
        return cv_index, offer_index, cv_embeddings_np, offer_embeddings_np
    
    def load_matching_data(self, use_sampled_data: bool = True) -> Tuple:
        """
        Charge les données et index pour le matching sémantique
        """
        if use_sampled_data:
            # Charger les données échantillonnées
            self.cv_dataset, self.offer_dataset, self.cv_id_to_index, self.offer_id_to_index = self.load_data_sampling()
            
            # Générer les embeddings
            self.cv_index, self.offer_index, self.cv_embeddings, self.offer_embeddings = self.generate_embeddings(
                self.cv_dataset, self.offer_dataset)
            
            return (self.cv_dataset, self.offer_dataset, self.cv_index, self.offer_index,
                    self.cv_embeddings, self.offer_embeddings, self.cv_id_to_index,
                    self.offer_id_to_index, self.sbert_model)
        else:
            # Utiliser les données complètes
            matching_dir = Path(self.config['data_paths']['matching_dir'])
            
            # Charger les datasets
            self.cv_dataset = pd.read_csv(matching_dir / 'cv_metadata.csv')
            self.offer_dataset = pd.read_csv(matching_dir / 'offer_metadata.csv')
            
            # Charger les index FAISS
            self.cv_index = faiss.read_index(str(matching_dir / 'cv_index.faiss'))
            self.offer_index = faiss.read_index(str(matching_dir / 'offer_index.faiss'))
            
            # Charger les embeddings
            self.cv_embeddings = np.load(matching_dir / 'cv_embeddings.npy')
            self.offer_embeddings = np.load(matching_dir / 'offer_embeddings.npy')
            
            # Charger les mappings ID->index
            with open(matching_dir / 'cv_id_mapping.json', 'r') as f:
                self.cv_id_to_index = json.load(f)
            with open(matching_dir / 'offer_id_mapping.json', 'r') as f:
                self.offer_id_to_index = json.load(f)
            
            # Charger le modèle SBERT pour le matching
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            return (self.cv_dataset, self.offer_dataset, self.cv_index, self.offer_index,
                    self.cv_embeddings, self.offer_embeddings, self.cv_id_to_index,
                    self.offer_id_to_index, self.sbert_model)
    
    def load_classification_models(self) -> Tuple:
        """Charge les modèles de classification pré-entraînés"""
        models_dir = Path(self.config['data_paths']['model_dir'])
        
        # Vérifier que le dossier existe
        if not models_dir.exists():
            raise FileNotFoundError(f"Le dossier {models_dir} n'existe pas")
        
        # Liste des fichiers requis
        required_files = [
            'CNN_final_model_classifier_resume.keras',
            'CNN_final_model_classifier_offer.keras',
            'label_encoder_resume.joblib',
            'label_encoder_offer.joblib'
        ]
        
        for file in required_files:
            if not (models_dir / file).exists():
                raise FileNotFoundError(f"Fichier manquant: {file}")
        
        # Chargement des modèles
        self.cv_classifier = load_model(models_dir / 'CNN_final_model_classifier_resume.keras')
        self.offer_classifier = load_model(models_dir / 'CNN_final_model_classifier_offer.keras')
        self.cv_le = joblib.load(models_dir / 'label_encoder_resume.joblib')
        self.offer_le = joblib.load(models_dir / 'label_encoder_offer.joblib')
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', from_pt=True)
        
        return self.cv_classifier, self.offer_classifier, self.cv_le, self.offer_le, self.tokenizer, self.distilbert_model
    
    def load_high_confidence_classes(self, confidence_threshold: float = 0.7) -> Tuple[List, List]:
        """
        Détermine les classes avec f1-score > seuil basé sur l'historique d'entraînement
        """
        models_dir = Path(self.config['data_paths']['model_dir'])
        
        try:
            # Charger les métriques d'évaluation
            cv_class_acc = pd.read_csv(models_dir / 'resume_class_metrics.csv')
            offer_class_acc = pd.read_csv(models_dir / 'offer_class_metrics.csv')
            
            # Filtrer les classes avec f1-score > threshold
            high_conf_cv_classes = cv_class_acc[cv_class_acc['f1_score'] > confidence_threshold]['class_name'].tolist()
            high_conf_offer_classes = offer_class_acc[offer_class_acc['f1_score'] > confidence_threshold]['class_name'].tolist()
            
        except FileNotFoundError:
            # Fallback: utiliser toutes les classes si les fichiers ne sont pas disponibles
            logger.warning("Fichiers de métriques non trouvés, utilisation de toutes les classes")
            self.load_classification_models()
            high_conf_cv_classes = self.cv_le.classes_.tolist()
            high_conf_offer_classes = self.offer_le.classes_.tolist()
        
        return high_conf_cv_classes, high_conf_offer_classes
    
    def create_manual_class_mapping(self) -> Dict[str, str]:
        """
        Crée un mapping manuel entre les classes de CVs et d'offres
        """
        manual_mapping = {
            # IT/Technologie - Correspondances exactes ou très similaires
            'Data Scientist': 'Data Scientist',
            'Software Developer': 'Software Developer',
            'Data Engineer': 'Data Engineer',
            'Web Developer': 'Web Developer',
            'DevOps/Cloud': 'Data Engineer',
            'Database Administrator': 'Database Administrator',
            'Network Administrator': 'Network Administrator',
            'Systems Administrator': 'Systems Administrator',
            'Security Analyst': 'Information Security Analyst',
            'Cyber Security Analyst': 'Information Security Analyst',
            'Information Security Analyst': 'Information Security Analyst',
            'Project Manager': 'Project Manager',
            'Software Engineer': 'Software Engineer',
            'AI/ML Specialist': 'Data Scientist',
            'Hardware Engineer': 'Hardware Engineer',
            'Data Analyst': 'Data Engineer',
            
            # Big Data & Cloud
            'Big Data Cloud Developer': 'Data Engineer',
            
            # Développeurs
            'Python Developer': 'Python / Java Developer',
            'Java Developer': 'Python / Java Developer',
            'Java Full Stack Developer': 'Python / Java Developer',
            'Python Developer,Software Developer': 'Python / Java Developer',
            'Java Developer,Software Developer': 'Python / Java Developer',
            'Front End Developer': 'Web Developer',
            'Full Stack Developer': 'Web Developer',
            'UI Developer': 'Web Developer',
            'Mobile Developer': 'Software Developer',
            'Java/J2Ee Developer': 'Python / Java Developer',
            'Senior Java Full Stack Developer': 'Python / Java Developer',
            'Senior Python Developer': 'Python / Java Developer',
            
            # Base de données
            'Oracle Database Administrator': 'Database Administrator',
            'SQL/SQL Server Database Administrator': 'Database Administrator',
            'Senior Database Administrator': 'Database Administrator',
            
            # Cybersécurité
            'Network Engineer': 'Network Administrator',
            
            # Management
            'It Manager': 'Other It Professional',
            'Consultant': 'Other It Professional',
            'Project Manager,Software Developer': 'Project Manager',
            
            # Catégories non-IT
            'Chef/Culinary': 'Chef/Culinary',
            'Mechanical Engineer': 'Mechanical Engineer',
            'Teacher/Professor': 'Teacher/Professor',
            'Legal Counsel': 'Legal Counsel',
            'HR Professional': 'HR Professional',
            'Financial Specialist': 'Financial Specialist',
            'Librarian/Archivist': 'Librarian/Archivist',
            'Research Scientist': 'Research Scientist',
            'Loan Officer': 'Loan Officer',
            'Bartender/Server': 'Bartender/Server',
            'Actuarial Specialist': 'Actuarial Specialist',
            'Counselor': 'Counselor',
            'Executive/C-Suite': 'Executive/C-Suite',
            
            # Catégories génériques
            'Other IT': 'Other It Professional',
            'Other No IT': 'Other No IT Professionnel',
            'Job Seeker': 'Other No IT Professionnel',
        }
        
        return manual_mapping
    
    def classify_text(self, text: str, text_type: str, classifier, le, tokenizer, 
                     distilbert_model, confidence_threshold: float = 0.7) -> Tuple[str, float, bool]:
        """
        Classifie un texte (CV ou offre) et retourne la classe prédite et le score de confiance
        """
        # Tokenization
        inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="tf")
        
        # Génération des embeddings avec DistilBERT
        outputs = distilbert_model(inputs)
        embeddings = outputs.last_hidden_state
        
        # Prédiction avec le modèle CNN
        predictions = classifier.predict(embeddings)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions)
        
        # Décodage de la classe
        predicted_class = le.inverse_transform([predicted_class_idx])[0]
        
        # Vérification du seuil de confiance
        is_high_confidence = confidence_score >= confidence_threshold
        
        return predicted_class, confidence_score, is_high_confidence
    
    def get_target_classes(self, source_class: str, source_type: str, manual_mapping: Dict[str, str]) -> List[str]:
        """
        Obtient les classes cibles basées sur le mapping manuel
        """
        target_classes = []
        
        if source_type == 'cv':
            # CV -> Offres: mapping direct
            if source_class in manual_mapping:
                target_classes.append(manual_mapping[source_class])
            else:
                # Si la classe n'est pas dans le mapping, utiliser la classe source comme fallback
                target_classes.append(source_class)
        else:
            # Offres -> CVs: mapping inverse
            target_classes = [cv_class for cv_class, offer_class in manual_mapping.items() 
                             if offer_class == source_class]
            
            # Si aucun mapping inverse trouvé, utiliser la classe source comme fallback
            if not target_classes:
                target_classes.append(source_class)
        
        return target_classes
    
    def filter_by_class(self, dataset: pd.DataFrame, target_classes: List[str], 
                       id_to_index: Dict, embeddings: np.ndarray, index) -> Tuple:
        """
        Filtre les embeddings et indices par classe cible
        """
        # Trouver les IDs des documents dans les classes cibles
        target_ids = dataset[dataset.iloc[:, 1].isin(target_classes)].iloc[:, 0].tolist()
        
        # Trouver les indices FAISS correspondants
        target_indices = [id_to_index[doc_id] for doc_id in target_ids if doc_id in id_to_index]
        
        if not target_indices:
            return None, None
        
        # Créer un sous-ensemble d'embeddings
        target_embeddings = embeddings[target_indices]
        
        # Créer un nouvel index FAISS pour le sous-ensemble
        dimension = target_embeddings.shape[1]
        target_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(target_embeddings)
        target_index.add(target_embeddings)
        
        return target_index, target_indices
    
    def hybrid_matching(self, query_text: str, query_type: str, top_k: int = 5, 
                       confidence_threshold: float = 0.7, use_sampled_data: bool = True) -> Tuple:
        """
        Fonction principale de matching hybride
        """
        # Charger les modèles et données
        (cv_classifier, offer_classifier, cv_le, offer_le,
         tokenizer, distilbert_model) = self.load_classification_models()
        
        (cv_dataset, offer_dataset, cv_index, offer_index,
         cv_embeddings, offer_embeddings, cv_id_to_index,
         offer_id_to_index, sbert_model) = self.load_matching_data(use_sampled_data=use_sampled_data)
        
        manual_mapping = self.create_manual_class_mapping()
        high_conf_cv_classes, high_conf_offer_classes = self.load_high_confidence_classes(confidence_threshold)
        
        # Déterminer le type de requête et configurer en conséquence
        if query_type == 'cv':
            classifier = cv_classifier
            le = cv_le
            high_conf_classes = high_conf_cv_classes
            target_data_type = 'offer'
            target_dataset = offer_dataset
            target_index_base = offer_index
            target_embeddings = offer_embeddings
            target_id_to_index = offer_id_to_index
        else:
            classifier = offer_classifier
            le = offer_le
            high_conf_classes = high_conf_offer_classes
            target_data_type = 'cv'
            target_dataset = cv_dataset
            target_index_base = cv_index
            target_embeddings = cv_embeddings
            target_id_to_index = cv_id_to_index
        
        # Étape 1: Classification de la requête
        predicted_class, confidence_score, is_high_confidence = self.classify_text(
            query_text, query_type, classifier, le, tokenizer, distilbert_model, confidence_threshold)
        
        # Afficher le résultat de classification
        self.display_classification_result(query_text, query_type, predicted_class, confidence_score)
        
        # Étape 2: Détermination des classes cibles
        if is_high_confidence and predicted_class in high_conf_classes:
            target_classes = self.get_target_classes(predicted_class, query_type, manual_mapping)
            print(f"➡️ Classes cibles recherchées : {target_classes}")
            
            # Étape 3: Filtrage par classes cibles
            # Vérifier si les classes cibles existent dans le dataset
            available_categories = target_dataset.iloc[:, 1].unique()
            existing_target_classes = [cls for cls in target_classes if cls in available_categories]
            
            if not existing_target_classes:
                print(f"⚠️ Avertissement: Aucune des classes cibles {target_classes} n'existe dans le dataset {target_data_type}")
                print(f"Classes disponibles: {list(available_categories)}")
                # Fallback: utiliser toutes les classes
                target_index = target_index_base
                target_indices = None
            else:
                # Utiliser seulement les classes qui existent
                target_index, target_indices = self.filter_by_class(
                    target_dataset, existing_target_classes, target_id_to_index,
                    target_embeddings, target_index_base)
        else:
            print("⚠️ Confiance insuffisante - recherche dans toutes les classes")
            target_index = target_index_base
            target_indices = None
        
        # Étape 4: Matching sémantique avec SBERT
        query_embedding = sbert_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Recherche des similarités
        if target_index is not None:
            distances, indices = target_index.search(query_embedding, top_k)
            
            # Préparation des résultats
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:  # Index invalide
                    continue
                
                # Si nous avons un sous-ensemble, mapper vers l'index original
                if target_indices is not None:
                    original_idx = target_indices[idx]
                else:
                    original_idx = idx
                
                # Récupérer les métadonnées du document
                if target_data_type == 'offer':
                    doc_id = target_dataset.iloc[original_idx]['offer_id']
                    doc_title = target_dataset.iloc[original_idx]['Job Title_Category']
                    doc_field = 'job_title'
                else:
                    doc_id = target_dataset.iloc[original_idx]['cv_id']
                    doc_title = target_dataset.iloc[original_idx]['Category']
                    doc_field = 'category'
                
                results.append({
                    'rank': i + 1,
                    'id': doc_id,
                    doc_field: doc_title,
                    'similarity_score': float(distance)
                })
            
            # Afficher les résultats de matching
            self.display_matching_results(results, query_type, predicted_class, confidence_score)
            
            return results, predicted_class, confidence_score
        else:
            print("\n❌ Aucun document trouvé dans les classes cibles spécifiées")
            print("📋 Vérification des catégories disponibles...")
            # Afficher les catégories disponibles pour diagnostic
            available_categories = target_dataset.iloc[:, 1].unique()
            print(f"Catégories disponibles : {list(available_categories)}")
            
            return [], predicted_class, confidence_score
    
    def display_classification_result(self, query_text: str, query_type: str, 
                                    predicted_class: str, confidence_score: float):
        """
        Affiche le résultat de la classification de manière claire et précise
        """
        # Extraire un ID à partir du texte (première partie du hash)
        query_id = self.generate_id(query_text)[:10]
        
        if query_type == 'cv':
            print(f"\n📄 CLASSIFICATION DU CV [{query_id}...]")
            print(f"   Ce CV est classifié comme : '{predicted_class}'")
        else:
            print(f"\n📋 CLASSIFICATION DE L'OFFRE [{query_id}...]")
            print(f"   Cette offre est classifiée comme : '{predicted_class}'")
        
        print(f"   Niveau de confiance : {confidence_score*100:.1f}%")
        
        # Interprétation du niveau de confiance
        if confidence_score >= 0.9:
            confidence_level = "Très élevée"
        elif confidence_score >= 0.7:
            confidence_level = "Élevée"
        elif confidence_score >= 0.5:
            confidence_level = "Modérée"
        else:
            confidence_level = "Faible"
        
        print(f"   Niveau de confiance : {confidence_level}")
    
    def display_matching_results(self, results: List[Dict], query_type: str, 
                               predicted_class: str, confidence_score: float):
        """
        Affiche les résultats de matching de manière claire et précise
        """
        if not results:
            print("\n❌ Aucune correspondance trouvée")
            return
        
        if query_type == 'cv':
            print(f"\n🎯 OFFRES CORRESPONDANTES POUR LE PROFIL '{predicted_class}'")
            print(f"   (Confiance de classification: {confidence_score*100:.1f}%)")
        else:
            print(f"\n👤 CVS CORRESPONDANTS POUR L'OFFRE '{predicted_class}'")
            print(f"   (Confiance de classification: {confidence_score*100:.1f}%)")
        
        print("   " + "="*60)
        
        for i, result in enumerate(results[:5]):  # Limiter aux 5 premiers résultats
            if 'job_title' in result:
                profile_info = f"{result['job_title']}"
                result_type = "Offre"
            else:
                profile_info = f"{result['category']}"
                result_type = "CV"
            
            # Formater le score de similarité
            similarity_score = result['similarity_score']
            if similarity_score >= 0.8:
                similarity_quality = "Excellente correspondance"
            elif similarity_score >= 0.6:
                similarity_quality = "Bonne correspondance"
            elif similarity_score >= 0.4:
                similarity_quality = "Correspondance moyenne"
            else:
                similarity_quality = "Faible correspondance"
            
            print(f"\n   {i+1}. {result_type} [{result['id'][:8]}...]")
            print(f"       Profil : {profile_info}")
            print(f"       Score de similarité : {similarity_score:.3f}")
            print(f"       Qualité : {similarity_quality}")
        
        # Afficher le meilleur résultat avec plus de détails
        best_result = results[0]
        if 'job_title' in best_result:
            best_profile = best_result['job_title']
            best_type = "Offre"
        else:
            best_profile = best_result['category']
            best_type = "CV"
        
        print(f"\n🏆 MEILLEURE CORRESPONDANCE")
        print(f"   {best_type} [{best_result['id']}]")
        print(f"   Profil : {best_profile}")
        print(f"   Score de similarité : {best_result['similarity_score']:.3f}")
    
    def evaluate_hybrid_system(self, test_cases: List[Tuple], confidence_threshold: float = 0.7, 
                             use_sampled_data: bool = True) -> List[Dict]:
        """
        Évalue le système hybride sur un ensemble de tests
        """
        print("🔍 ÉVALUATION DU SYSTÈME HYBRIDE")
        print("="*80)
        
        results = []
        
        for i, (query_text, query_type, expected_class) in enumerate(test_cases):
            print(f"\nTest {i+1}/{len(test_cases)}: {query_type.upper()}")
            print(f"Texte: {query_text[:100]}...")
            
            # Exécuter le matching hybride
            matching_results, predicted_class, confidence = self.hybrid_matching(
                query_text, query_type, top_k=3, confidence_threshold=confidence_threshold, 
                use_sampled_data=use_sampled_data)
            
            # Évaluer la classification
            classification_correct = (predicted_class == expected_class)
            classification_confidence = confidence
            
            # Évaluer le matching (simplifié)
            has_results = len(matching_results) > 0
            top_score = matching_results[0]['similarity_score'] if has_results else 0
            
            results.append({
                'test_id': i + 1,
                'query_type': query_type,
                'expected_class': expected_class,
                'predicted_class': predicted_class,
                'classification_correct': classification_correct,
                'classification_confidence': classification_confidence,
                'has_matching_results': has_results,
                'top_matching_score': top_score
            })
            
            print(f"Résultat: {'✓' if classification_correct else '✗'} {predicted_class} "
                  f"(confiance: {confidence:.3f}, matching: {top_score:.3f})")
        
        # Calcul des métriques
        total_tests = len(results)
        correct_classifications = sum(1 for r in results if r['classification_correct'])
        classification_accuracy = correct_classifications / total_tests
        avg_confidence = np.mean([r['classification_confidence'] for r in results])
        avg_matching_score = np.mean([r['top_matching_score'] for r in results if r['has_matching_results']])
        
        print(f"\n{'='*80}")
        print("📊 RÉSULTATS DE L'ÉVALUATION")
        print(f"{'='*80}")
        print(f"Précision de classification: {classification_accuracy:.3f}")
        print(f"Confiance moyenne: {avg_confidence:.3f}")
        print(f"Score de matching moyen: {avg_matching_score:.3f}")
        
        return results
    
    def display_evaluation_results(self, evaluation_results: List[Dict]):
        """
        Affiche les résultats de l'évaluation de manière claire
        """
        print("\n" + "="*80)
        print("📈 RÉSULTATS DE L'ÉVALUATION DU SYSTÈME")
        print("="*80)
        
        # Calcul des métriques
        total_tests = len(evaluation_results)
        correct_classifications = sum(1 for r in evaluation_results if r['classification_correct'])
        classification_accuracy = correct_classifications / total_tests
        avg_confidence = np.mean([r['classification_confidence'] for r in evaluation_results])
        avg_matching_score = np.mean([r['top_matching_score'] for r in evaluation_results if r['has_matching_results']])
        success_rate = sum(1 for r in evaluation_results if r['has_matching_results']) / total_tests
        
        print(f"• Précision de classification : {classification_accuracy*100:.1f}%")
        print(f"• Confiance moyenne : {avg_confidence*100:.1f}%")
        print(f"• Score de matching moyen : {avg_matching_score:.3f}")
        print(f"• Taux de succès (au moins une correspondance) : {success_rate*100:.1f}%")
        
        # Détail par test
        print(f"\n• DÉTAIL DES TESTS ({total_tests} tests effectués)")
        print(" " + "-"*70)
        
        for result in evaluation_results:
            status_icon = "✅" if result['classification_correct'] else "❌"
            match_icon = "✓" if result['has_matching_results'] else "✗"
            print(f"   {status_icon} Test {result['test_id']}: {result['query_type'].upper()}")
            print(f"       Attendu: {result['expected_class']}, Prédit: {result['predicted_class']}")
            print(f"       Confiance: {result['classification_confidence']*100:.1f}%")
            print(f"       Matching: {match_icon} (meilleur score: {result['top_matching_score']:.3f})")
            print()


def main():
    """Fonction principale"""
    # Initialiser le matcher
    matcher = HybridMatcher()
    
    print("🤖 SYSTÈME DE MATCHING CV-OFFRES AVEC CLASSIFICATION HYBRIDE")
    print("="*80)
    
    # Test avec un CV
    sample_cv = "Data analyst with 5 years of experience in SQL, Python, and data visualization. Strong analytical skills and experience with Tableau and Power BI."
    results, predicted_class, confidence = matcher.hybrid_matching(sample_cv, "cv", top_k=5, use_sampled_data=True)
    
    # Test avec une offre
    sample_offer = "Looking for a data scientist with machine learning experience and Python programming skills. Knowledge of TensorFlow or PyTorch is a plus."
    results, predicted_class, confidence = matcher.hybrid_matching(sample_offer, "offer", top_k=5, use_sampled_data=True)
    
    # Évaluation du système
    test_cases = [
        ("Data analyst with SQL and Python experience", "cv", "Data Analyst"),
        ("Software developer with Java and Spring framework", "cv", "Software Developer"),
        ("Data scientist position with machine learning requirements", "offer", "Data Scientist"),
        ("Web developer with React and JavaScript", "offer", "Web Developer")
    ]
    
    evaluation_results = matcher.evaluate_hybrid_system(test_cases, use_sampled_data=True)
    matcher.display_evaluation_results(evaluation_results)


if __name__ == "__main__":
    main()