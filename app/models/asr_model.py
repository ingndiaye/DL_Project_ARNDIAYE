
import torch
import sys
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.config import Config

class ASRModel:
    def __init__(self, model_name=Config.ASR_MODEL_NAME):
        self.device = self._get_device()
        print(f" Device: {self.device}")
        print(f" Chargement du modèle ASR...")
        
        #  Utiliser le cache local
        cache_dir = str(Config.MODELS_CACHE_DIR / "asr")
        print(f" Cache: {cache_dir}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        try:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_safetensors=True
            )
        except:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_safetensors=False
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.sample_rate = Config.SAMPLE_RATE
        print(f" ASR chargé depuis {cache_dir}")
    
    def _get_device(self):
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def transcribe(self, audio_array):
        if audio_array is None or len(audio_array) == 0:
            raise ValueError("Audio array vide")
        
        inputs = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') else None
        
        with torch.no_grad():
            logits = self.model(
                input_values,
                attention_mask=attention_mask
            ).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription.strip()