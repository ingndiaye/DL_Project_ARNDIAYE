# app/space_app.py
import gradio as gr
import tempfile
import os
import shutil
from app.utils.audio_processor import AudioProcessor
from app.models.asr_model import ASRModel
from app.models.sentiment_model import SentimentModel
from app.config import Config

# Chargement des modèles (une seule fois)
print(" Chargement des modèles pour Hugging Face Space...")
audio_processor = AudioProcessor()
asr_model = ASRModel()
sentiment_model = SentimentModel()
print(" Modèles chargés avec succès!")

def analyze_audio(audio_file):
    """
    Fonction principale pour l'interface Gradio sur Hugging Face Spaces
    """
    if audio_file is None:
        return "Veuillez télécharger un fichier audio", "", 0.0
    
    try:
        # Vérifier l'extension
        file_ext = os.path.splitext(audio_file)[1].lower()
        if file_ext not in Config.SUPPORTED_FORMATS:
            return f" Format non supporté. Utilisez: {Config.SUPPORTED_FORMATS}", "", 0.0
        
        # Prétraitement
        audio_array = audio_processor.load_audio(audio_file)
        
        # Transcription
        transcription = asr_model.transcribe(audio_array)
        
        # Analyse de sentiment
        sentiment, confidence = sentiment_model.predict(transcription)
        
        # Emojis pour le sentiment
        emoji_map = {
            'positif': '😊',
            'negatif': '😞',
            'neutre': '😐'
        }
        
        result_text = f"""
        ## 📊 Résultat de l'analyse
        
        ### Sentiment : {emoji_map.get(sentiment, '')} **{sentiment.upper()}**
        ### Confiance : **{confidence:.2%}**
        
        ###  Transcription :
        > {transcription}
        """
        
        return result_text, transcription, confidence
        
    except ValueError as e:
        return f" Erreur: {str(e)}", "", 0.0
    except Exception as e:
        return f" Erreur inattendue: {str(e)}", "", 0.0

# Interface Gradio optimisée pour HF Spaces
def create_space_interface():
    with gr.Blocks(
        title="🎤 Analyse de Sentiment Vocal",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px;
            margin: auto;
        }
        """
    ) as demo:
        gr.Markdown("""
        #  Analyse de Sentiment par Voix
        
        **Transformez vos appels vocaux en insights émotionnels !**
        
        Téléchargez un fichier audio et obtenez instantanément la transcription 
        et l'analyse de sentiment.
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="📁 Télécharger un fichier audio",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                analyze_btn = gr.Button("🔍 Analyser", variant="primary", size="lg")
                
                gr.Markdown("""
                ###  Formats supportés
                - `.wav` et `.mp3`
                - Durée maximale: **5 minutes**
                
                ### Classes de sentiment
                - 😊 **Positif** (satisfait)
                - 😞 **Négatif** (mécontent)
                - 😐 **Neutre**
                """)
            
            with gr.Column(scale=2):
                sentiment_output = gr.Markdown(label="📊 Résultat")
                transcription_output = gr.Textbox(
                    label=" Transcription complète",
                    lines=8,
                    placeholder="La transcription apparaîtra ici..."
                )
                confidence_output = gr.Number(
                    label="🎯 Score de confiance",
                    precision=4,
                    interactive=False
                )
        
        # Lier le bouton
        analyze_btn.click(
            fn=analyze_audio,
            inputs=[audio_input],
            outputs=[sentiment_output, transcription_output, confidence_output]
        )
        
        # Auto-analyse quand un fichier est uploadé
        audio_input.change(
            fn=analyze_audio,
            inputs=[audio_input],
            outputs=[sentiment_output, transcription_output, confidence_output]
        )
        
        gr.Markdown("""
        ---
        ###  Modèles utilisés
        - **ASR**: Wav2Vec 2.0 (jonatasgrosman/wav2vec2-large-xlsr-53-french)
        - **Sentiment**: DistilBERT Emotion (bhadresh-savani/distilbert-base-uncased-emotion)
         
        """)
        
    return demo

# Point d'entrée pour Hugging Face Spaces
if __name__ == "__main__":
    demo = create_space_interface()
    demo.launch()