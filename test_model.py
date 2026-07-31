# test_sentiment.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.sentiment_model import SentimentModel

def test_sentiment():
    print("=" * 60)
    print("🧪 TEST DU MODÈLE DE SENTIMENT")
    print("=" * 60)
    
    model = SentimentModel()
    
    test_phrases = [
        "Je suis très satisfait du service, merci beaucoup!",
        "Excellent travail, je recommande vivement!",
        "Je suis déçu par la qualité du produit.",
        "Service client horrible, je ne reviendrai plus.",
        "Je voudrais des informations sur mon compte."
    ]
    
    print("\n📝 Résultats:")
    print("-" * 60)
    
    for phrase in test_phrases:
        sentiment, confidence = model.predict(phrase)
        emoji = "😊" if sentiment == "positif" else "😞" if sentiment == "négatif" else "😐"
        print(f"{emoji} {sentiment.upper():8} ({confidence:.1%}) - {phrase[:50]}...")
    
    print("=" * 60)

if __name__ == "__main__":
    test_sentiment()