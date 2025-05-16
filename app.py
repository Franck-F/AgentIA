import streamlit as st
import time
import arxiv
from dotenv import load_dotenv
import os
import re
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

# --- Configuration Google API Key ---
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Clé API Google (GOOGLE_API_KEY) non trouvée. Veuillez la configurer dans le fichier .env ou les variables d'environnement.")
    st.stop()

# --- Initialisation du client Gemini ---
try:
    # Utiliser le modèle Gemini souhaité (ex: gemini-1.5-pro)
    gemini_client = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du modèle Gemini: {e}")
    st.stop()

# Initialisation des variables de session
if "start_chat" not in st.session_state:
    st.session_state.start_chat = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    # Utiliser le format Langchain pour les messages
    st.session_state.messages = []

st.set_page_config(page_title="Bac2MyFuture", page_icon=":book:", layout="wide")

# Fonction pour nettoyer la réponse de l'IA (peut rester si nécessaire)
def clean_response(response):
    # Ajuster si le format des sources change avec Gemini
    return re.sub(r' [\d+:\d+source]', " ", response)

# Fonction pour rechercher sur ArXiv
def search_arxiv(query):
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in search.results():
        summary = ' '.join(result.summary.split()[:100]) + '...'
        results.append({
            'title': result.title,
            'summary': summary,
            'authors': [author.name for author in result.authors],
            'url': result.entry_id
        })
    return results

# Fonction principale pour gérer la conversation avec Gemini
def conversation_chat(query):
    # Ajouter le message utilisateur au format Langchain
    st.session_state.messages.append(HumanMessage(content=query))

    with st.spinner("L'assistant réfléchit..."):
        try:
            # Appeler le client Gemini avec l'historique des messages
            response = gemini_client.invoke(st.session_state.messages)
            assistant_reply = response.content
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API Gemini: {e}")
            assistant_reply = "Désolé, une erreur est survenue lors de la communication avec l'assistant."

    # Ajouter la réponse de l'assistant au format Langchain
    st.session_state.messages.append(AIMessage(content=assistant_reply))
    return assistant_reply

# Initialisation de l'état de session pour l'affichage (
def initialize_session_state_display():
    # Utiliser directement st.session_state.messages pour l'affichage
    pass 

# Affichage de l'historique du chat depuis st.session_state.messages
def display_chat_history():
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content) 
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content) 
       

# --- Interface principale de l'application ---

st.title("Bacc2Future 🎯 ")
st.write("Bienvenue sur Bac2Futur, un assistant pour vous aider à trouver la formation et l'orientation qui vous convient.")

# Bouton pour démarrer/réinitialiser
if st.sidebar.button("Démarrer/Réinitialiser la conversation 💬 🚀💻"):
    st.session_state.start_chat = True
    st.session_state.messages = [] 
    st.rerun() 

# Affichage de l'historique et gestion de la conversation
if st.session_state.start_chat:
    display_chat_history()

    if user_input := st.chat_input("Posez votre question ici..."):
        conversation_chat(user_input)
        st.rerun()
else:
    st.info("Cliquez sur 'Démarrer/Réinitialiser la conversation' dans la barre latérale pour commencer.")

# --- Code ArXiv (si toujours nécessaire) ---
st.sidebar.header("Recherche ArXiv")
arxiv_query = st.sidebar.text_input("Rechercher sur ArXiv")
if st.sidebar.button("Lancer la recherche"):
    if arxiv_query:
        st.sidebar.subheader("Résultats ArXiv:")
        arxiv_results = search_arxiv(arxiv_query)
        if arxiv_results:
            for res in arxiv_results:
                st.sidebar.markdown(f"**[{res['title']}]({res['url']})**")
                st.sidebar.caption(f"Auteurs: {', '.join(res['authors'])}")
                st.sidebar.markdown(f"_{res['summary']}_")
        else:
            st.sidebar.info("Aucun résultat trouvé sur ArXiv.")
    else:
        st.sidebar.warning("Veuillez entrer un terme de recherche.")

