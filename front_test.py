import streamlit as st
import requests
import os

# Configuration de l'URL du backend
BACKEND_API_URL = "http://127.0.0.1:8000"
CHAT_ENDPOINT = f"{BACKEND_API_URL}/chat"

st.set_page_config(page_title="Bac2Futur", page_icon=":mortar_board:", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_assistant_response(user_query: str):
    if not user_query:
        return "Veuillez entrer une question."
    
    payload = {"query": user_query}
    try:
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Désolé, je n'ai pas pu obtenir de réponse.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de communication avec le backend : {e}")
        return "Impossible de contacter l'assistant pour le moment. Veuillez réessayer plus tard."
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {e}")
        return "Une erreur s'est produite."

title_container = st.container()
chat_container = st.container()

with title_container:
    st.title("Bac2Futur 🎯")
    st.write("Bienvenue sur Bac2Futur, un assistant pour vous aider à trouver la formation et l'orientation qui vous convient.")

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Posez votre question ici...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("L'assistant réfléchit..."):
            assistant_reply = get_assistant_response(user_input)
            message_placeholder.markdown(assistant_reply)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
