
import sys
import os
import base64
from pathlib import Path

# Ajouter le chemin parent avant tout autre import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from app.utils.audio_processor import AudioProcessor
from app.models.asr_model import ASRModel
from app.models.sentiment_model import SentimentModel
from app.config import Config

# ============================================
# CHARGEMENT DU CSS EXTERNE
# ============================================
def load_css():
    """Charger le CSS depuis le fichier static/style.css"""
    css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f" Fichier CSS non trouvé: {css_path}")
        return ""

# ============================================
# LOGO EN BASE64
# ============================================
def get_logo_base64():
    """Convertir le logo en base64 pour l'intégrer directement."""
    logo_path = os.path.join(os.path.dirname(__file__), "static", "dit-logo.png")
    
    try:
        with open(logo_path, "rb") as f:
            logo_data = f.read()
            logo_base64 = base64.b64encode(logo_data).decode()
            return f'data:image/png;base64,{logo_base64}'
    except FileNotFoundError:
        print(f" Logo non trouvé: {logo_path}")
        return None

# ============================================
# CHARGEMENT DES MODÈLES
# ============================================
print("🚀 Chargement des modèles...")
audio_processor = AudioProcessor()
asr_model = ASRModel()
sentiment_model = SentimentModel()
print(" Modèles chargés avec succès!")

# ============================================
# FONCTION D'ANALYSE
# ============================================
def analyze_audio(audio_file):
    """
    Fonction principale pour l'interface Gradio.
    """
    if audio_file is None:
        return """
        <div class="info-message">
            📤 <strong>Téléchargez un fichier audio</strong><br>
            ou utilisez le microphone
        </div>
        """, "", 0.0
    
    try:
        # Vérifier l'extension
        file_ext = os.path.splitext(audio_file)[1].lower()
        if file_ext not in Config.SUPPORTED_FORMATS:
            return f"""
            <div class="error-message">
                 <strong>Format non supporté</strong><br>
                Utilisez: {Config.SUPPORTED_FORMATS}
            </div>
            """, "", 0.0
        
        # Prétraitement
        audio_array = audio_processor.load_audio(audio_file)
        
        # Transcription
        transcription = asr_model.transcribe(audio_array)
        
        # Analyse de sentiment
        sentiment, confidence = sentiment_model.predict(transcription)
        
        # Emojis et couleurs
        emoji_map = {
            'positif': '😊',
            'negatif': '😞',
            'neutre': '😐'
        }
        
        label_class = {
            'positif': 'label-positif',
            'negatif': 'label-negatif',
            'neutre': 'label-neutre'
        }
        
        confidence_percent = confidence * 100
        
        # Résultat HTML stylisé
        result_html = f"""
        <div class="result-card">
            <h3>Résultat de l'analyse</h3>
            
            <div class="result-sentiment">
                <div class="result-emoji">{emoji_map.get(sentiment, '😐')}</div>
                <div class="result-label {label_class.get(sentiment, 'label-neutre')}">
                    {sentiment.upper()}
                </div>
                <div class="result-confidence">
                    Confiance: <strong>{confidence_percent:.1f}%</strong>
                </div>
            </div>
            
            <div class="result-transcription">
                <strong>Transcription :</strong><br>
                "{transcription}"
            </div>
            
            <div style="text-align:right; font-size:11px; opacity:0.6; margin-top:10px;">
                 {os.path.basename(audio_file)}
            </div>
        </div>
        """
        
        return result_html, transcription, confidence
        
    except ValueError as e:
        return f"""
        <div class="error-message">
             <strong>Erreur:</strong> {str(e)}
        </div>
        """, "", 0.0
    except Exception as e:
        return f"""
        <div class="error-message">
             <strong>Erreur inattendue:</strong> {str(e)}
        </div>
        """, "", 0.0

# ============================================
# CRÉATION DE L'INTERFACE GRADIO
# ============================================
def create_gradio_interface():
    """Créer l'interface Gradio avec design amélioré."""
    
    # Charger le CSS externe
    custom_css = load_css()
    
    # Obtenir le logo en base64
    logo_base64 = get_logo_base64()
    
    #  Si le logo existe, utiliser l'image, sinon utiliser du texte
    if logo_base64:
        logo_html = f'<img src="{logo_base64}" alt="Logo DIT" style="width:55px; height:55px; object-fit:contain;">'
    else:
        logo_html = """
        <div style="text-align:center; line-height:1.1;">
            <div style="font-size:20px; font-weight:bold; color:#667eea;">DIT</div>
            <div style="font-size:7px; color:#888; text-transform:uppercase;">
                DAKAR<br>INSTITUTE OF<br>TECHNOLOGY
            </div>
        </div>
        """
    
    with gr.Blocks(
        title="Analyse de Sentiment Vocal - DIT",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
        ),
        css=custom_css,
        elem_id="main-container"
    ) as demo:
        
        # ============================================
        # EN-TÊTE AVEC LOGO
        # ============================================
        gr.HTML(f"""
        <div class="header-section">
            <div class="header-left">
                <div class="logo-container">
                    {logo_html}
                </div>
                <div>
                    <div class="header-title">EXAMEN DL 2 - ANALYSE DE SENTIMENT</div>
                    <div class="header-subtitle">Détection automatique de sentiment dans des appels vocaux</div>
                    <div class="header-subtitle">Rèalisè par ABDOURAHMANE NDIAYE</div>
                </div>
            </div>        
        </div>
        """)
        
       
        
        # ============================================
        # CORPS PRINCIPAL
        # ============================================
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Charger un fichier")
                
                audio_input = gr.File(
                    label="",
                    file_types=[".wav", ".mp3"],
                    type="filepath"
                )
                
                gr.Markdown("###  Utiliser le microphone")
                
                 
                try:
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label=""
                    )
                except TypeError:
                    try:
                        mic_input = gr.Audio(
                            source="microphone",
                            type="filepath",
                            label=""
                        )
                    except:
                        mic_input = gr.Audio(
                            type="filepath",
                            label="",
                            interactive=True
                        )
                
                analyze_btn = gr.Button(
                    " Analyser",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ###  Informations
                - **Formats**: `.wav`, `.mp3`
                - **Durée max**: 5 minutes
                - **Classes**: Positif, Négatif, Neutre
                """)
            
            with gr.Column(scale=2):
                sentiment_output = gr.HTML(
                    label=" Résultat",
                    value="""
                    <div class="info-message">
                         <strong>Chargez un fichier audio</strong><br>
                        ou utilisez le microphone 
                    </div>
                    """
                )
                
                transcription_output = gr.Textbox(
                    label=" Transcription",
                    lines=4,
                    placeholder="La transcription apparaîtra ici...",
                    interactive=False
                )
                
                confidence_output = gr.Number(
                    label=" Score de confiance",
                    precision=4,
                    value=None,
                    interactive=False
                )
        
        # ============================================
        # LIEN DES BOUTONS
        # ============================================
        analyze_btn.click(
            fn=analyze_audio,
            inputs=[audio_input],
            outputs=[sentiment_output, transcription_output, confidence_output]
        )
        
        # Lier le microphone
        try:
            mic_input.change(
                fn=analyze_audio,
                inputs=[mic_input],
                outputs=[sentiment_output, transcription_output, confidence_output]
            )
        except:
            pass
        
        # ============================================
        # PIED DE PAGE AVEC COORDONNÉES
        # ============================================
        gr.HTML("""
        <div class="footer-section">
            <div class="footer-title"> Dakar Institute of Technology (DIT)</div>
            <div class="footer-subtitle">Module: Deep Learning 2 | Projet d'Examen 2026</div>
            
            <div class="footer-info">
                <div class="footer-subtitle"><strong> Étudiant:</strong> ABDOURAHMANE NDIAYE</div>
                <div class="footer-subtitle"><strong> Email:</strong> ingndiaye@gmail.com</div>
                <div class="footer-subtitle"><strong> Téléphone:</strong> +221 77 921 20 45</div>
            </div>
             
        </div>
        """)
        
    return demo

# ============================================
# POINT D'ENTRÉE
# ============================================
if __name__ == "__main__":
    demo = create_gradio_interface()
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="purple",
            )
        )
    except TypeError:
        try:
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True
            )
        except:
            demo.launch(share=True)