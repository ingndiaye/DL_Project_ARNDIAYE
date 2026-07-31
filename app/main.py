
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path

# Imports avec "app." pour fonctionner partout
from app.utils.audio_processor import AudioProcessor
from app.models.asr_model import ASRModel
from app.models.sentiment_model import SentimentModel
from app.config import Config

# Initialisation
app = FastAPI(
    title="Speech Sentiment Analysis API",
    description="Analyse de sentiment à partir d'appels vocaux",
    version="1.0.0"
)

# Chargement des modèles (lazy loading)
audio_processor = None
asr_model = None
sentiment_model = None

def load_models():
    global audio_processor, asr_model, sentiment_model
    if audio_processor is None:
        audio_processor = AudioProcessor()
        asr_model = ASRModel()
        sentiment_model = SentimentModel()
    return audio_processor, asr_model, sentiment_model

@app.get("/")
async def root():
    return {"message": "API de détection de sentiment par voix", "status": "ready"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": all([audio_processor, asr_model, sentiment_model])}

@app.post("/predict")
async def predict_sentiment(file: UploadFile = File(...)):
    """
    Endpoint principal pour l'analyse de sentiment.
    Accepte un fichier audio et retourne la transcription et le sentiment.
    """
    try:
        # Charger les modèles
        ap, asr, sent = load_models()
        
        # Vérifier le format du fichier
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in Config.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté. Utilisez: {Config.SUPPORTED_FORMATS}"
            )
        
        # Sauvegarder temporairement le fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Fichier vide")
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Prétraitement audio
            audio_array = ap.load_audio(tmp_path)
            
            # Transcription
            transcription = asr.transcribe(audio_array)
            
            # Analyse de sentiment
            sentiment, confidence = sent.predict(transcription)
            
            # Résultat
            return JSONResponse({
                "success": True,
                "transcription": transcription,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "file_name": file.filename,
                "duration_seconds": len(audio_array) / Config.SAMPLE_RATE
            })
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inattendue: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Endpoint pour l'analyse de plusieurs fichiers audio.
    """
    results = []
    for file in files:
        try:
            result = await predict_sentiment(file)
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "file_name": file.filename,
                "error": str(e)
            })
    return JSONResponse({"results": results})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)