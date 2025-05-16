import streamlit as st
import re
import os
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import ChatPromptTemplate, RunnablePassthrough
from langchain.chains import LLMChain
from langchain.retrievers import MultiQueryRetriever
from langchain.prompts import PromptTemplate


OPENAI_API_KEY = 'APIkey'

# Configuration de la page
st.set_page_config(page_title="Bac2Futur", page_icon=":mortar_board:", layout="wide")

# Initialisation de la base de données vectorielle et du modèle
vector_db = Chroma(persist_directory="./chroma_db")
llm = ChatOpenAI(model_name="gpt-4o")
def load_and_index_pdfs(directory):
    loader = DirectoryLoader(directory, loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return db

# Chargement des documents
pdf_directory = "data/pdfs"  # Dossier contenant les PDFs
if os.path.exists(pdf_directory):
    vector_db = load_and_index_pdfs(pdf_directory)
else:
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    
    
# Définition du prompt pour l'orientation
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    Tu es un assistant spécialisé en orientation académique et professionnelle en France.
    Commence par recueillir les informations suivantes auprès de l'utilisateur :
    - Ses intérêts et compétences.
    - Ses préférences d'études (universitaire, alternance, etc.).
    - Son niveau d'études actuel et ses aspirations professionnelles.
    Une fois ces informations collectées, oriente-le en fonction des données disponibles.
    Si une information dépasse ta connaissance, réponds : "Je ne dispose pas d'informations à ce sujet."
    Question: {question}
    """
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

template = """Réponds en utilisant uniquement ce contexte :
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Fonction pour gérer la conversation
def conversation_chat(query):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("L'assistant réfléchit..."):
        response = chain.invoke({"question": query})

    st.session_state.messages.append({"role": "assistant", "content": response})
    return response

# Initialisation de l'état de session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Interface principale
title_container = st.container()
chat_container = st.container()

with title_container:
    st.title("Bac2Futur 🎯")
    st.write("Bienvenue sur Bac2Futur, un assistant pour vous aider à trouver la formation et l'orientation qui vous convient.")

# Affichage de la conversation
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"], avatar_style="thumbs")
        else:
            st.chat_message("assistant").write(msg["content"], avatar_style="fun-emoji")

    user_input = st.text_input("Posez votre question :", placeholder="Tapez ici...")
    if user_input:
        response = conversation_chat(user_input)
        st.chat_message("assistant").write(response, avatar_style="fun-emoji")
