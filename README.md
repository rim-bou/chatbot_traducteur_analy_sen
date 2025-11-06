Chatbot Multilingue avec Traduction et Analyse de Sentiment



\##  Description

Ce projet est une application conversationnelle développée avec \*\*Streamlit\*\* et les modèles de \*\*Hugging Face\*\* :

\- 💬 Chatbot : `microsoft/DialoGPT-medium`

\- 🌍 Traduction : `Helsinki-NLP/opus-mt-fr-en`

\- 🧠 Analyse de sentiment : `nlptown/bert-base-multilingual-uncased-sentiment`



L'utilisateur peut discuter avec le chatbot, voir la traduction anglaise de chaque message et son sentiment (positif, neutre, négatif).



---



\##  Installation



\### 1️⃣ Créer un environnement virtuel



```bash

python -m venv .venv

.venv\\Scripts\\activate



"PowerShell



.\\.venv\\Scripts\\Activate.ps1



"invite de commande

.venv\\Scripts\\activate



\###Installation des depandances



créer un fichier requirement.txt avec le contenu suivant



transformer

torch

streamlit

sentencepiece





puis installer toute les bibliothèques

pip install -r requirement.txt





Cette commande lit ton fichier et installe automatiquement :



transformers → modèles Hugging Face



torch → moteur PyTorch



streamlit → interface web



sentencepiece → nécessaire à la traduction





Détails techniques

Modèles utilisés (Hugging Face)



Rôle	Nom du modèle	Description

&nbsp;Chatbot	microsoft/DialoGPT-medium	Répond aux messages en anglais

&nbsp;Traduction	Helsinki-NLP/opus-mt-fr-en	Traduit le français vers l’anglais

&nbsp;Sentiment	nlptown/bert-base-multilingual-uncased-sentiment





Lance l'application

streamlit run chatbot\_streamlit.py



Cela ouvre automatiquement une page web à l’adresse :

http://localhost:8501



### 

### 











