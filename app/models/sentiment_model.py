
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.config import Config

class SentimentModel:
    def __init__(self, model_name=Config.SENTIMENT_MODEL_NAME):
        self.device = self._get_device()
        print(f" Device: {self.device}")
        print(f" Chargement du modèle de sentiment: {model_name}")
        
        cache_dir = str(Config.MODELS_CACHE_DIR / "sentiment") if hasattr(Config, 'MODELS_CACHE_DIR') else None
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_safetensors=True
            )
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_safetensors=False
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Mapping 5 étoiles → 3 classes
        # 0: 1 star, 1: 2 stars, 2: 3 stars, 3: 4 stars, 4: 5 stars
        self.label_map = {
            0: "négatif",    # 1 étoile → négatif
            1: "négatif",    # 2 étoiles → négatif
            2: "neutre",     # 3 étoiles → neutre
            3: "positif",    # 4 étoiles → positif
            4: "positif"     # 5 étoiles → positif
        }
        
        print(f" Modèle de sentiment chargé avec succès!")
        print(f"   Mapping: 0-1→négatif, 2→neutre, 3-4→positif")
    
    def _get_device(self):
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def predict(self, text):
        if not text or len(text.strip()) == 0:
            return "neutre", 0.0
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask
            )
        
        probabilities = F.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)
        
        confidence = float(confidence.cpu().numpy())
        predicted_class = int(predicted_class.cpu().numpy())
        
        sentiment = self.label_map.get(predicted_class, "neutre")
        
        # Ajuster la confiance
        if sentiment == "négatif":
            confidence = confidence * 0.8
        elif sentiment == "positif":
            confidence = confidence * 0.9
        
        if confidence < 0.5:
            sentiment = "neutre"
            confidence = 1 - confidence
            
        return sentiment, confidence