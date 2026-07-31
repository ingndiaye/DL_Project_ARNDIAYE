# download_models.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC,
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from app.config import Config

def download_models():
    print("=" * 60)
    print("🚀 TÉLÉCHARGEMENT DES MODÈLES")
    print("=" * 60)
    
    # Créer le dossier de cache
    Config.MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. ASR Model (Wav2Vec2)
    print("\n📥 Téléchargement du modèle ASR (Wav2Vec2)...")
    asr_cache = Config.MODELS_CACHE_DIR / "asr"
    asr_cache.mkdir(exist_ok=True)
    
    try:
        processor = Wav2Vec2Processor.from_pretrained(
            Config.ASR_MODEL_NAME,
            cache_dir=str(asr_cache)
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            Config.ASR_MODEL_NAME,
            cache_dir=str(asr_cache),
            use_safetensors=True
        )
        print(f"✅ ASR téléchargé dans {asr_cache}")
    except Exception as e:
        print(f"❌ Erreur ASR: {e}")
    
    # 2. Sentiment Model (CamemBERT)
    print("\n📥 Téléchargement du modèle de sentiment (CamemBERT)...")
    sentiment_cache = Config.MODELS_CACHE_DIR / "sentiment"
    sentiment_cache.mkdir(exist_ok=True)
    
    try:
        # ✅ Télécharger avec les bonnes options
        tokenizer = AutoTokenizer.from_pretrained(
            Config.SENTIMENT_MODEL_NAME,
            cache_dir=str(sentiment_cache),
            use_fast=True  # Utiliser le tokenizer rapide
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            Config.SENTIMENT_MODEL_NAME,
            cache_dir=str(sentiment_cache),
            use_safetensors=True
        )
        print(f"✅ CamemBERT téléchargé dans {sentiment_cache}")
        
    except Exception as e:
        print(f"⚠️ Erreur avec use_fast=True, tentative avec use_fast=False...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                Config.SENTIMENT_MODEL_NAME,
                cache_dir=str(sentiment_cache),
                use_fast=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                Config.SENTIMENT_MODEL_NAME,
                cache_dir=str(sentiment_cache),
                use_safetensors=True
            )
            print(f"✅ CamemBERT téléchargé (use_fast=False) dans {sentiment_cache}")
        except Exception as e2:
            print(f"❌ Erreur Sentiment: {e2}")
    
    # Afficher la taille
    print("\n" + "=" * 60)
    total_size = sum(f.stat().st_size for f in Config.MODELS_CACHE_DIR.rglob('*') if f.is_file())
    print(f"📊 Taille totale des modèles: {total_size / (1024**2):.2f} Mo")
    print("=" * 60)

if __name__ == "__main__":
    download_models()