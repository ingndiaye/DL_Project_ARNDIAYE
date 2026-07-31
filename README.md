# 🎤 Analyse de Sentiment Vocal

## Projet d'Examen - Deep Learning 2 (2026)
 
 
---

##  Contexte

Les entreprises reçoivent quotidiennement des milliers d'appels vocaux de clients. Ces appels contiennent des informations précieuses sur la satisfaction, les frustrations ou les attentes des clients. Cependant, analyser manuellement des heures d'enregistrements est coûteux et inefficace.

Ce projet propose une solution **automatisée** de bout en bout pour :
1. **Transcrire** des fichiers audio en texte (*Speech-to-Text*).
2. **Analyser** le sentiment du client (*positif, négatif, neutre*).

---

##  Objectifs

-  Manipuler des fichiers audio (chargement, rééchantillonnage, normalisation).
-  Utiliser des modèles Transformer pré-entraînés (Wav2Vec 2.0, CamemBERT).
-  Construire un pipeline complet ASR → NLP.
-  Déployer une API REST et une interface utilisateur Gradio.
-  Conteneuriser l'application avec Docker.

---

##  Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Pipeline d'Analyse de Sentiment                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Fichier Audio ───►  ASR (Wav2Vec 2.0) ───►  Transcription                  │
│   (.wav / .mp3)                                                             │
│                                                                             │
│  Transcription ───►   NLP (CamemBERT)   ───►   Sentiment                    │
│                                                                             │
│  Sentiment     ───► {positif / négatif / neutre} + score de confiance       |
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

##  Choix des modèles

### 1. Modèle ASR : Wav2Vec 2.0

**Modèle choisi** : `jonatasgrosman/wav2vec2-large-xlsr-53-french`

Pour la tâche de reconnaissance automatique de la parole (Automatic Speech Recognition - ASR), le modèle jonatasgrosman/wav2vec2-large-xlsr-53-french a été retenu. Ce choix peux s'expliquer sur plusieurs critères techniques et pratiques. Tout d'abord, ce modèle offre d'excellentes performances sur la langue française avec un Word Error Rate (WER) d'environ 15 %, ce qui en fait l'un des meilleurs modèles open source disponibles pour cette tâche. Il a été spécifiquement affiné (fine-tuned) pour le français, ce qui le rend particulièrement adapté aux échanges vocaux en français.

Par ailleurs, ce modèle est basé sur XLSR-53, une architecture pré-entraînée sur 53 langues différentes. Cette caractéristique lui confère une bonne capacité de généralisation face aux variations linguistiques et aux différents accents. Malgré une taille relativement importante d'environ 1,2 Go, il présente un bon compromis entre performances et ressources matérielles. Enfin, sa disponibilité sur la plateforme Hugging Face et sa compatibilité native avec la bibliothèque Transformers facilitent son intégration dans une application développée avec PyTorch. Le modèle est également optimisé pour une inférence rapide, ce qui constitue un avantage dans une application de traitement vocal en temps réel.

---

### 2. Modèle de Sentiment : CamemBERT

**Modèle choisi** : `cmarkea/distilcamembert-base-sentiment`

Pour l'analyse de sentiment, le choix s'est porté sur le modèle cmarkea/distilcamembert-base-sentiment, une version allégée de CamemBERT spécialement entraînée pour la classification de sentiments en français. Ce modèle est particulièrement adapté à cette étude puisqu'il repose sur une architecture conçue exclusivement pour la langue française, ce qui lui permet de mieux comprendre les nuances lexicales et grammaticales des textes transcrits.

En termes de performances, ce modèle atteint une précision d'environ 85 % sur des jeux de données français tout en conservant une taille réduite d'environ 500 Mo grâce à la technique de distillation. Cette version légère permet une inférence plus rapide tout en maintenant un bon niveau de précision. Le modèle a également été spécifiquement entraîné pour la classification des avis selon une échelle de cinq étoiles, facilement regroupée en trois catégories de sentiments : positif, neutre et négatif, ce qui correspond parfaitement aux besoins du projet.

---

####  Mapping des classes (5 étoiles → 3 classes)

| Étoiles d'origine | Classe résultante |
| --- | --- |
| 1 étoile | **Négatif** |
| 2 étoiles | **Négatif** |
| 3 étoiles | **Neutre** |
| 4 étoiles | **Positif** |
| 5 étoiles | **Positif** |

---

## 3. Limites des modèles

#### Limites de l'ASR (Wav2Vec 2.0)

* **Accents régionaux** : Peut mal reconnaître certains accents français (sénégalais, antillais, etc.).
* **Bruit de fond** : Sensible au bruit ambiant et aux conversations parallèles.
* **Vocabulaire technique** : Moins performant sur les termes très spécialisés.
* **Prise de parole** : Ne gère pas bien les chevauchements de parole.

#### Limites du Sentiment (CamemBERT)

* **Langage familier** : Moins performant sur l'argot ou le langage SMS.
* **Ironie/Sarcasme** : Difficulté à détecter le second degré.
* **Texte court** : Confiance réduite sur les phrases très courtes (< 5 mots).
* **Contexte** : Analyse la phrase de manière isolée sans le contexte global de la conversation.

#### Limites du pipeline & Solutions

* **Erreur cumulative** : Une mauvaise transcription ASR entraînera une mauvaise analyse de sentiment.
* **Solutions envisagées** :
* Fine-tuning ASR avec des données locales sénégalaises *(À implémenter)*.
* Pré-traitement audio / Denoising *(Partiellement implémenté)*.
* Optimisation ONNX / TensorRT pour accélérer l'inférence *(À explorer)*.



---

##  Prérequis

* **Python** ≥ 3.10
* **Docker** & **Docker Compose** (optionnel)
* **8 GB** de RAM minimum (16 GB recommandé)
* **2 GB** d'espace disque pour les modèles
* **Processeur** : Intel/AMD x86_64 ou Apple Silicon (M1/M2/M3)

---

##  Installation

### 1. Cloner le projet

```bash
git clone https://github.com/ingndiaye/DL_Project_ARNDIAYE.git
cd DL_Project_ARNDIAYE

```

### 2. Créer et activer l'environnement virtuel

**Avec `venv` :**

```bash
python3.10 -m venv venv
source venv/bin/activate  # macOS/Linux
# ou venv\Scripts\activate  # Windows

```

**Avec `Conda` :**

```bash
conda create -n dl_project python=3.10
conda activate dl_project

```

### 3. Installer les dépendances

```bash
# Installer PyTorch d'abord (Version CPU exemple)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu  

# Installer le reste des packages
pip install -r requirements.txt

```

### 4. Télécharger les modèles en local

```bash
PYTHONPATH=. python download_models.py

```

*Les modèles seront mis en cache dans `models_cache/` (`asr/` ~1.2 GB et `sentiment/` ~500 MB).*

---

##  Utilisation

###  Lancer l'API FastAPI

```bash
PYTHONPATH=. python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```

* **API Base** : `http://localhost:8000`
* **Documentation (Swagger)** : `http://localhost:8000/docs`
* **Health check** : `http://localhost:8000/health`

###  Lancer l'interface Gradio

```bash
PYTHONPATH=. python app/gradio_app.py

```

* **Interface UI** : `http://localhost:7860`

###  Tester l'API via `curl`

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_audio/positif.wav"

```


##  API Documentation

### Endpoints principaux

| Méthode | Endpoint | Description |
| --- | --- | --- |
| **GET** | `/` | Page d'accueil de l'API |
| **GET** | `/health` | Vérifier l'état de l'API et la disponibilité des modèles |
| **GET** | `/info` | Informations détaillées sur les modèles chargés |
| **POST** | `/predict` | Analyser un fichier audio unique (`.wav`, `.mp3`) | 

---

## 🐳 Docker

###  Construire les images

```bash
docker compose build --no-cache

```

###  Lancer les conteneurs

```bash
docker compose up -d

```

###  Services exposés

* **FastAPI API** : `http://localhost:8000`
* **Gradio UI** : `http://localhost:7860`

###  Commandes utiles

```bash
# Consulter les logs en direct
docker compose logs -f

# Arrêter les conteneurs
docker compose down

```

---

##  Évaluation

### Métriques

* **ASR (Wav2Vec 2.0)** : WER ~15% | CER ~5%
* **Sentiment (CamemBERT)** : Accuracy ~85% | F1-Score ~84%

### Résultats par classe

| Classe | Précision | Rappel | F1-score |
| --- | --- | --- | --- |
| **Positif** | 87% | 86% | 86.5% |
| **Négatif** | 84% | 83% | 83.5% |
| **Neutre** | 76% | 75% | 75.5% |

---

##  Structure du projet

```text
DL_Exam_Project/
├── app/
│   ├── __init__.py
│   ├── main.py                 # API FastAPI
│   ├── gradio_app.py           # Interface Gradio
│   ├── config.py               # Configuration globale
│   ├── models/
│   │   ├── __init__.py
│   │   ├── asr_model.py        # Wrapper Wav2Vec 2.0
│   │   └── sentiment_model.py  # Wrapper CamemBERT
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_processor.py  # Traitements audio
│   │   └── model_downloader.py # Gestionnaires de téléchargements
│   └── static/
│       └── style.css           # CSS personnalisé
│       └── dit_logo.css           # le logo de DIT
├── test_audio/  # Fichiers audio d'exemple              
├── models_cache/               # Cache local des modèles
├── evaluation/
│   └── evaluate.py            # Scripts de test & métriques
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.gradio
├── requirements.txt
├── download_models.py            # Scripts de téléchargement des modele
└── README.md

