import argparse
from models.classification_resume import main as run_resume_classification
from models.classification_offer import main as run_offer_classification

def main():
    parser = argparse.ArgumentParser(description='Exécuter les scripts de classification')
    parser.add_argument('--resume', action='store_true', help='Exécuter la classification des CV')
    parser.add_argument('--offer', action='store_true', help='Exécuter la classification des offres')
    parser.add_argument('--all', action='store_true', help='Exécuter toutes les classifications')
    
    args = parser.parse_args()
    
    # Exécution des scripts
    if args.resume or args.all:
        print("Début de la classification des CV...")
        try:
            run_resume_classification()
            print("Classification des CV terminée avec succès")
        except Exception as e:
            print(f"Échec de la classification des CV: {e}")
    
    if args.offer or args.all:
        print("Début de la classification des offres...")
        try:
            run_offer_classification()
            print("Classification des offres terminée avec succès")
        except Exception as e:
            print(f"Échec de la classification des offres: {e}")

if __name__ == "__main__":
    main()