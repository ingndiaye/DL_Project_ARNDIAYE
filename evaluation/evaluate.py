
import os
import json
import torch
import numpy as np
from pathlib import Path
from jiwer import wer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from app.utils.audio_processor import AudioProcessor
from app.models.asr_model import ASRModel
from app.models.sentiment_model import SentimentModel
from app.config import Config

class PipelineEvaluator:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.asr_model = ASRModel()
        self.sentiment_model = SentimentModel()
        
    def evaluate_asr(self, test_data_path):
        """
        Évaluer la qualité de la transcription (WER)
        test_data_path: chemin vers un fichier JSON avec les transcriptions de référence
        """
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        transcriptions = []
        references = []
        errors = []
        
        for item in test_data:
            try:
                audio_path = item['audio_path']
                reference = item['reference_text']
                
                # Transcrire
                audio_array = self.audio_processor.load_audio(audio_path)
                transcription = self.asr_model.transcribe(audio_array)
                
                transcriptions.append(transcription)
                references.append(reference)
                
                print(f" {Path(audio_path).name}")
                print(f"   Référence: {reference}")
                print(f"   Prédiction: {transcription}")
                
            except Exception as e:
                errors.append({
                    'audio': audio_path,
                    'error': str(e)
                })
                print(f" Erreur sur {Path(audio_path).name}: {e}")
        
        # Calcul du WER
        if transcriptions:
            wer_score = wer(references, transcriptions)
            print(f"\n WER (Word Error Rate): {wer_score:.2%}")
            return {
                'wer': wer_score,
                'transcriptions': transcriptions,
                'references': references,
                'errors': errors
            }
        return None
    
    def evaluate_sentiment(self, test_data_path):
        """
        Évaluer la performance du modèle de sentiment (Accuracy/F1)
        test_data_path: chemin vers un fichier JSON avec les sentiments de référence
        """
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        predictions = []
        references = []
        confidences = []
        
        for item in test_data:
            try:
                text = item['text']
                reference = item['sentiment']  # 'positif', 'negatif', 'neutre'
                
                # Prédire
                sentiment, confidence = self.sentiment_model.predict(text)
                
                predictions.append(sentiment)
                references.append(reference)
                confidences.append(confidence)
                
                print(f" Texte: {text[:50]}...")
                print(f"   Référence: {reference} | Prédiction: {sentiment} (conf: {confidence:.2%})")
                
            except Exception as e:
                print(f" Erreur: {e}")
        
        # Calcul des métriques
        if predictions:
            accuracy = accuracy_score(references, predictions)
            
            # F1 score (weighted pour gérer les classes déséquilibrées)
            f1 = f1_score(references, predictions, average='weighted', labels=['positif', 'negatif', 'neutre'])
            
            # Rapport détaillé
            report = classification_report(
                references, 
                predictions, 
                labels=['positif', 'negatif', 'neutre'],
                target_names=['Positif', 'Négatif', 'Neutre']
            )
            
            print(f"\n Accuracy: {accuracy:.2%}")
            print(f" F1-score (weighted): {f1:.2%}")
            print("\n Rapport détaillé:")
            print(report)
            
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predictions,
                'references': references,
                'confidences': confidences,
                'report': report
            }
        return None
    
    def evaluate_full_pipeline(self, audio_data_path, sentiment_data_path):
        """
        Évaluer tout le pipeline complet
        """
        print("=" * 60)
        print(" ÉVALUATION DU PIPELINE COMPLET")
        print("=" * 60)
        
        print("\n 1. Évaluation de l'ASR (WER)")
        print("-" * 40)
        asr_results = self.evaluate_asr(audio_data_path)
        
        print("\n 2. Évaluation du Sentiment (Accuracy/F1)")
        print("-" * 40)
        sentiment_results = self.evaluate_sentiment(sentiment_data_path)
        
        # Résumé final
        print("\n" + "=" * 60)
        print(" RÉSUMÉ DES PERFORMANCES")
        print("=" * 60)
        
        summary = {
            'asr': asr_results,
            'sentiment': sentiment_results,
            'timestamp': datetime.now().isoformat()
        }
        
        if asr_results:
            print(f"\n ASR - WER: {asr_results['wer']:.2%}")
        if sentiment_results:
            print(f" Sentiment - Accuracy: {sentiment_results['accuracy']:.2%}")
            print(f" Sentiment - F1-score: {sentiment_results['f1_score']:.2%}")
        
        # Sauvegarder les résultats
        output_path = Path('evaluation/results.json')
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n Résultats sauvegardés dans: {output_path}")
        return summary

if __name__ == "__main__":
    from datetime import datetime
    
    evaluator = PipelineEvaluator()
    
    
    ASR_TEST_DATA = "evaluation/data/asr_test.json"
    SENTIMENT_TEST_DATA = "evaluation/data/sentiment_test.json"
    
    evaluator.evaluate_full_pipeline(ASR_TEST_DATA, SENTIMENT_TEST_DATA)