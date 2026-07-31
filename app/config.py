import os
from pathlib import Path

class Config:
    # Audio parameters
    SAMPLE_RATE = 16000
    MAX_DURATION_SECONDS = 300  # 5 minutes
    SUPPORTED_FORMATS = ['.wav', '.mp3']
    
    # Model parameters
    ASR_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    # SENTIMENT_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"  # Pour l'anglais
    # Alternative pour le français:
    SENTIMENT_MODEL_NAME = "cmarkea/distilcamembert-base-sentiment"
    
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    TEST_AUDIO_DIR = BASE_DIR / "tests" / "test_audio"
    # MODELS CACHE - Dans votre projet
    MODELS_CACHE_DIR = BASE_DIR / "models_cache"
    

    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    #API_HOST = "0.0.0.0"
    #API_PORT = 8000
