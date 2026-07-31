# app/utils/audio_processor.py
import sys
import os

 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import librosa
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from app.config import Config

class AudioProcessor:
    def __init__(self, target_sr=Config.SAMPLE_RATE, max_duration=Config.MAX_DURATION_SECONDS):
        self.target_sr = target_sr
        self.max_duration = max_duration
        
    def load_audio(self, audio_path):
        """Charger et prétraiter un fichier audio."""
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
        
        file_ext = Path(audio_path).suffix.lower()
        if file_ext not in Config.SUPPORTED_FORMATS:
            raise ValueError(f"Format non supporté. Utilisez: {Config.SUPPORTED_FORMATS}")
        
        try:
            # Utiliser librosa pour charger l'audio
            if file_ext == '.wav':
                waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            else:  # mp3
                # Utiliser pydub pour mp3
                audio = AudioSegment.from_mp3(audio_path)
                audio_path_wav = str(Path(audio_path).with_suffix('.temp.wav'))
                audio.export(audio_path_wav, format="wav")
                waveform, sr = librosa.load(audio_path_wav, sr=self.target_sr, mono=True)
                Path(audio_path_wav).unlink()
                
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement audio: {e}")
        
        # Vérifier la durée
        duration = len(waveform) / sr
        if duration > self.max_duration:
            raise ValueError(f"Audio trop long: {duration:.2f}s (max: {self.max_duration}s)")
        
        if len(waveform) == 0:
            raise ValueError("Fichier audio vide")
        
        return self.preprocess_audio(waveform)
    
    def preprocess_audio(self, waveform):
        """Prétraiter l'audio: normalisation."""
        # Normalisation de l'amplitude
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        # Vérifier le silence
        if np.mean(np.abs(waveform)) < 0.001:
            raise ValueError("Audio trop silencieux ou vide")
        
        return waveform