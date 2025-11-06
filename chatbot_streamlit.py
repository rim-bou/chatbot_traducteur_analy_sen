import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer, pipeline

# --- Configuration de la page ---
st.set_page_config(page_title="Chatbot ", page_icon="🤖")
st.title("🤖 Chatbot  avec traduction en anglais et analyse de sentiment")

st.markdown("""
Cette application combine **3 modèles d'intelligence artificielle** :
- 💬 Chatbot : `microsoft/DialoGPT-medium`
- 🌍 Traduction : `Helsinki-NLP/opus-mt-fr-en`
- 🧠 Analyse de sentiment : `nlptown/bert-base-multilingual-uncased-sentiment`
""")

# --- Chargement des modèles ---
@st.cache_resource
def load_models():
    # Chatbot
    chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Traduction FR → EN
    trans_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    trans_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    # Analyse de sentiment
    sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    return chat_tokenizer, chat_model, trans_tokenizer, trans_model, sentiment

chat_tokenizer, chat_model, trans_tokenizer, trans_model, sentiment_analyzer = load_models()

# --- Fonctions utilitaires ---
def traduire_fr_en(texte):
    inputs = trans_tokenizer([texte], return_tensors="pt", padding=True)
    translated = trans_model.generate(**inputs)
    return trans_tokenizer.decode(translated[0], skip_special_tokens=True)

def analyser_sentiment(texte):
    result = sentiment_analyzer(texte)[0]["label"]
    if "1" in result or "2" in result:
        return "😡 Négatif"
    elif "3" in result:
        return "😐 Neutre"
    else:
        return "😊 Positif"

# --- Zone de saisie utilisateur ---
user_input = st.text_input("💬 Vous :", "")

if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("Envoyer"):
    if user_input.strip() == "":
        st.warning("⚠️ Veuillez entrer un message.")
    else:
        # 1️⃣ Traduire le texte en anglais
        english_input = traduire_fr_en(user_input)
        # 2️⃣ Analyse du sentiment
        sentiment_user = analyser_sentiment(user_input)
        # 3️⃣ Générer une réponse du chatbot
        inputs = chat_tokenizer.encode(english_input + chat_tokenizer.eos_token, return_tensors="pt")
        outputs = chat_model.generate(inputs, max_length=1000, pad_token_id=chat_tokenizer.eos_token_id)
        response_english = chat_tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        # 4️⃣ Traduire la réponse en français
        back_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        back_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        inputs_back = back_tokenizer([response_english], return_tensors="pt", padding=True)
        translated_back = back_model.generate(**inputs_back)
        response_french = back_tokenizer.decode(translated_back[0], skip_special_tokens=True)
        # 5️⃣ Sentiment de la réponse
        sentiment_bot = analyser_sentiment(response_french)

        # Stocker l’échange
        st.session_state["history"].append({
            "user": user_input,
            "user_en": english_input,
            "user_sent": sentiment_user,
            "bot": response_french,
            "bot_en": response_english,
            "bot_sent": sentiment_bot
        })

# --- Affichage de l’historique ---
for msg in st.session_state["history"]:
    st.markdown(f"### 👤 Utilisateur : {msg['user']}")
    st.write(f"**Traduction anglaise :** {msg['user_en']}")
    st.write(f"**Sentiment :** {msg['user_sent']}")
    st.markdown(f"### 🤖 Chatbot : {msg['bot']}")
    st.write(f"**Traduction anglaise :** {msg['bot_en']}")
    st.write(f"**Sentiment :** {msg['bot_sent']}")
    st.divider()
