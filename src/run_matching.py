"""
Script principal pour exécuter le système de matching CV-offres
"""

import argparse
from models.matching_model import HybridMatcher

def main():
    parser = argparse.ArgumentParser(description='Système de matching CV-offres')
    parser.add_argument('--text', type=str, help='Texte à matcher (CV ou offre)')
    parser.add_argument('--type', type=str, choices=['cv', 'offer'], help='Type de texte (cv ou offer)')
    parser.add_argument('--top_k', type=int, default=5, help='Nombre de résultats à retourner')
    parser.add_argument('--eval', action='store_true', help='Exécuter le mode évaluation')
    
    args = parser.parse_args()
    
    # Initialiser le matcher
    matcher = HybridMatcher()
    
    if args.eval:
        # Mode évaluation
        test_cases = [
            ("Data analyst with SQL and Python experience", "cv", "Data Analyst"),
            ("Software developer with Java and Spring framework", "cv", "Software Developer"),
            ("Data scientist position with machine learning requirements", "offer", "Data Scientist"),
            ("Web developer with React and JavaScript", "offer", "Web Developer")
        ]
        
        evaluation_results = matcher.evaluate_hybrid_system(test_cases, use_sampled_data=True)
        matcher.display_evaluation_results(evaluation_results)
    elif args.text and args.type:
        # Mode matching simple
        results, predicted_class, confidence = matcher.hybrid_matching(
            args.text, args.type, top_k=args.top_k, use_sampled_data=True
        )
    else:
        print("Usage: python run_matching.py --text <texte> --type <cv|offer> [--top_k 5]")
        print("   ou: python run_matching.py --eval")

if __name__ == "__main__":
    main()