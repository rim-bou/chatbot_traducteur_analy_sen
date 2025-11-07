import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer, pipeline
import torch

# --- Configuration de la page ---
st.set_page_config(page_title="Chatbot ", page_icon="🤖")
st.title("🤖 Chatbot  avec Traduction  français anglais et Analyse de Sentiment")

st.markdown("""
Ce chatbot te permet de discuter en français 💬  
Il traduit automatiquement tes messages et les réponses en anglais   
et analyse aussi les émotions exprimées 😄😐😡
---
""")

# --- Chargement des modèles (mise en cache) ---
@st.cache_resource
def load_models():
    chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    trans_fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    trans_fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    trans_en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    trans_en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    return chat_tokenizer, chat_model, trans_fr_en_tokenizer, trans_fr_en_model, trans_en_fr_tokenizer, trans_en_fr_model, sentiment_analyzer


chat_tokenizer, chat_model, trans_fr_en_tokenizer, trans_fr_en_model, trans_en_fr_tokenizer, trans_en_fr_model, sentiment_analyzer = load_models()


# --- Fonctions utilitaires ---
def traduire_fr_en(texte):
    inputs = trans_fr_en_tokenizer([texte], return_tensors="pt", padding=True)
    translated = trans_fr_en_model.generate(**inputs)
    return trans_fr_en_tokenizer.decode(translated[0], skip_special_tokens=True)

def traduire_en_fr(texte):
    inputs = trans_en_fr_tokenizer([texte], return_tensors="pt", padding=True)
    translated = trans_en_fr_model.generate(**inputs)
    return trans_en_fr_tokenizer.decode(translated[0], skip_special_tokens=True)

def analyser_sentiment(texte):
    result = sentiment_analyzer(texte)[0]["label"]
    if "1" in result or "2" in result:
        return "😡 Négatif"
    elif "3" in result:
        return "😐 Neutre"
    else:
        return "😊 Positif"

# --- Initialisation de la session ---
if "chat_history_ids" not in st.session_state:
    st.session_state["chat_history_ids"] = None
if "past_inputs" not in st.session_state:
    st.session_state["past_inputs"] = []
if "generated_responses" not in st.session_state:
    st.session_state["generated_responses"] = []

# --- Zone de saisie ---
st.subheader("💬 Conversation")
user_input = st.chat_input("Écris ton message ici...")

# --- Bouton de réinitialisation ---
if st.sidebar.button("🗑️ Réinitialiser la conversation"):
    st.session_state["chat_history_ids"] = None
    st.session_state["past_inputs"] = []
    st.session_state["generated_responses"] = []
    st.sidebar.success("Conversation réinitialisée ✅")

# --- Si l'utilisateur envoie un message ---
if user_input:
    with st.spinner("Le bot réfléchit... 🤔"):
        # 1️⃣ Traduction du message utilisateur (FR -> EN)
        user_en = traduire_fr_en(user_input)

        # 2️⃣ Encodage + historique
        new_input_ids = chat_tokenizer.encode(user_en + chat_tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([st.session_state["chat_history_ids"], new_input_ids], dim=-1) if st.session_state["chat_history_ids"] is not None else new_input_ids

        # 3️⃣ Génération de la réponse du chatbot
        chat_history_ids = chat_model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=chat_tokenizer.eos_token_id
        )
        st.session_state["chat_history_ids"] = chat_history_ids

        # 4️⃣ Décodage et traduction inverse (EN -> FR)
        bot_output_en = chat_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        bot_output_fr = traduire_en_fr(bot_output_en)

        # 5️⃣ Analyse des sentiments
        user_sent = analyser_sentiment(user_input)
        bot_sent = analyser_sentiment(bot_output_fr)

        # 6️⃣ Ajout à l'historique
        st.session_state["past_inputs"].append((user_input, user_en, user_sent))
        st.session_state["generated_responses"].append((bot_output_fr, bot_output_en, bot_sent))

# --- Affichage de la conversation ---
if st.session_state["generated_responses"]:
    for i in range(len(st.session_state["generated_responses"])):
        user_fr, user_en, user_sent = st.session_state["past_inputs"][i]
        bot_fr, bot_en, bot_sent = st.session_state["generated_responses"][i]

        with st.chat_message("user"):
            st.markdown(f"**{user_fr}**")
            st.caption(f"🇬🇧 Traduction anglaise : *{user_en}*")
            st.caption(f"Sentiment : {user_sent}")

        with st.chat_message("assistant"):
            st.markdown(f"**{bot_fr}**")
            st.caption(f"🇬🇧 Traduction anglaise : *{bot_en}*")
            st.caption(f"Sentiment : {bot_sent}")
