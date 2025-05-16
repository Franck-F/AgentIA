import os
import re
import streamlit as st
import chromadb
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from rank_bm25 import BM25Okapi
from langchain.schema import BaseRetriever
from typing import List
from langchain.schema import Document
from pydantic import Field, PrivateAttr

# Configuration API OpenAI
OPENAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialisation de ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("rag_chatbot")

# chargement et indexation des données 
def load_and_index_pdfs(directory):
    loader = DirectoryLoader(directory, loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1000,
        embedding_ctx_length=8191
    )
    db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return db

# Chargement des documents
pdf_directory = "data/pdfs"  
if os.path.exists(pdf_directory):
    vector_db = load_and_index_pdfs(pdf_directory)
else:
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())

# Définition du prompt
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Rôle & Mission
Tu es un assistant virtuel spécialisé dans l’orientation académique et professionnelle des élèves et étudiants en France.
Ton objectif est de les guider dans :
• Le choix de leurs études et de leur carrière en fonction de leurs intérêts et compétences.
• L’analyse des offres de formation et d’emploi pour identifier celles qui leur correspondent le mieux.
• L’accompagnement dans la rédaction de documents essentiels : 
  o Lettres de motivation.
  o Projets de formation motivés (notamment pour Parcoursup).
  o CV et autres documents de candidature.
• Le suivi des candidatures, en aidant à organiser un journal de candidatures et à optimiser leurs démarches.
Tu dois toujours fournir des informations précises, adaptées et à jour, exclusivement pour la France.

Méthodologie de réponse
Lorsqu’un utilisateur pose une question, tu dois :
1. Comprendre précisément la demande et poser des questions de clarification.
2. Fournir une réponse claire, détaillée et adaptée au niveau et au contexte de l’utilisateur.
3. S’adapter au ton de l’utilisateur.
4. Structurer tes réponses pour une meilleure lisibilité.
5. Ne pas inventer d’informations.

Exemples de demandes
• "Peux-tu m'expliquer comment fonctionne Parcoursup ?"
• "Comment structurer une lettre de motivation efficace ?"
• "Quelle est la rémunération d’un apprenti en France ?"
• "propose moi des formations après le baccalauréat "


Limites et restrictions
1. Pas de conseils médicaux ou juridiques.
2. Pas d’événements récents au-delà de la date de mise à jour.
3. Pas d’opinions personnelles ou d’émotions.

Question: {question}""",
)

# Création d'un retriever personnalisé avec BM25
class BM25EnhancedRetriever(BaseRetriever):
    _vectorstore: object = PrivateAttr()
    _k: int = PrivateAttr()
    _base_retriever: object = PrivateAttr()
    
    def __init__(self, vectorstore, k: int = 4):
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k
        self._base_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        # Récupérer les documents via le retriever de base
        docs = self._base_retriever.get_relevant_documents(query)
        
        if not docs:
            return []
            
        # Préparer les documents pour BM25
        corpus = [doc.page_content for doc in docs]
        tokenized_corpus = [doc.split() for doc in corpus]
        
        # Créer et entraîner le modèle BM25
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Tokeniser la requête
        tokenized_query = query.split()
        
        # Calculer les scores BM25
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Trier les documents selon les scores BM25
        scored_docs = list(zip(docs, bm25_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les k meilleurs documents
        return [doc for doc, score in scored_docs[:self._k]]
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# retriever avec BM25
retriever = MultiQueryRetriever.from_llm(
    BM25EnhancedRetriever(vector_db),
    ChatOpenAI(),
    prompt=QUERY_PROMPT
)

# RAG prompt template
template = """Répond à la question en utilisant uniquement le contexte suivant:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Création de la chaîne RAG
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()

)

# Fonction pour détecter si une question de clarification est nécessaire
def needs_clarification(query):
    llm = ChatOpenAI(temperature=0.7)
    clarification_prompt = ChatPromptTemplate.from_template("""
    Analyse la question suivante et détermine si elle nécessite des clarifications.
    Question: {query}
     
    Réponds par "True" si la question est vague ou nécessite plus de contexte,
    "False" si la question est claire et spécifique.
    Réponse (True/False uniquement):
    """)
     
    response = llm(clarification_prompt.format_messages(query=query))
    return response.content.strip().lower() == "true"

# Fonction pour générer des questions de clarification
def generate_clarifying_questions(query):
    llm = ChatOpenAI(temperature=0.7)
    question_prompt = ChatPromptTemplate.from_template("""
    En tant qu'assistant d'orientation, génère 2-3 questions pertinentes pour mieux comprendre le contexte de la demande suivante:
    {query}
    
    Format: Retourne uniquement les questions, une par ligne.
    """)
    
    response = llm(question_prompt.format_messages(query=query))
    return response.content.strip().split('\n')

# Fonction pour nettoyer la réponse
def clean_response(response):
    if not response or response.strip() == "":
        return "Je ne trouve pas de réponse pertinente à votre question. Pourriez-vous la reformuler ou me donner plus de détails ?"
    return response.strip()

# Fonction pour détecter les formules de politesse
def detect_greeting(text):
    greetings = {
        "bonjour": "Bonjour",
        "bonsoir": "Bonsoir",
        "salut": "Salut",
        "hey": "Hey",
        "coucou": "Coucou",
        "hello": "Hello"
    }
    
    first_words = text.lower().split()[:2]
    for word in first_words:
        if word in greetings:
            return greetings[word]
    return None

# Fonction pour générer une réponse complète
def generate_complete_response(query, conversation_history=None):
    llm = ChatOpenAI(temperature=0.7)
    
    # Construire le contexte à partir de l'historique des conversations
    context = ""
    if conversation_history:
        # Prendre les 5 derniers échanges pour garder le contexte pertinent
        recent_history = conversation_history[-5:]
        context = "\n".join([
            f"{'User' if i%2==0 else 'Assistant'}: {msg}" 
            for i, msg in enumerate(recent_history)
        ])
    
    # Créer un prompt qui inclut l'historique et guide la réponse
    response_prompt = ChatPromptTemplate.from_template("""
    Tu es un assistant spécialisé dans l'orientation scolaire et professionnelle.
    
    Historique de la conversation:
    {context}
    
    Question actuelle: {query}
    
    En tenant compte de l'historique de la conversation et de la question actuelle:
    1. Analyse le contexte global de la discussion
    2. Assure-toi que ta réponse est cohérente avec les informations précédemment données
    3. Fais référence aux éléments pertinents des échanges précédents si nécessaire
    4. Si la question actuelle contredit ou remet en question des informations précédentes, 
       explique poliment la contradiction et clarifie la situation
    
    Réponds de manière claire, précise et personnalisée:
    """)
    
    # Générer la réponse en utilisant le contexte
    messages = response_prompt.format_messages(
        context=context if context else "Pas d'historique disponible",
        query=query
    )
    response = llm(messages)
    
    # Nettoyer et retourner la réponse
    return {
        "response": clean_response(response.content),
        "needs_clarification": False
    }

# Fonction principale de conversation
def conversation_chat(query):
    try:
        # Vérifier les salutations
        greeting = detect_greeting(query)
        if greeting and len(query.split()) <= 2:
            return {
                "response": f"{greeting}, comment puis-je vous aider ?",
                "needs_clarification": False
            }

        # Gérer les précisions
        if st.session_state.awaiting_clarification and not needs_clarification(query):
            st.session_state.clarification_context.append(query)
            
            # Générer la réponse finale avec le contexte
            final_response = generate_complete_response(
                st.session_state.original_query,
                st.session_state.conversation_history
            )
            
            # Réinitialiser l'état
            st.session_state.awaiting_clarification = False
            st.session_state.original_query = None
            st.session_state.clarification_context = []
            
            return {
                "response": final_response["response"],
                "needs_clarification": False
            }

        # Nouvelle question nécessitant des précisions
        if needs_clarification(query):
            st.session_state.awaiting_clarification = True
            st.session_state.original_query = query
            clarifying_questions = generate_clarifying_questions(query)
            return {
                "response": "Pour mieux vous aider, j'aurais besoin de quelques précisions :\n" + "\n".join(clarifying_questions),
                "needs_clarification": True
            }

        # Question directe
        response_data = generate_complete_response(query, st.session_state.conversation_history)
        if greeting:
            response_data["response"] = f"{greeting}! {response_data['response']}"
        return response_data

    except Exception as e:
        return {
            "response": f"Je suis désolé, mais je rencontre une difficulté : {str(e)}",
            "needs_clarification": False
        }

# Initialisation des variables de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "original_query" not in st.session_state:
    st.session_state.original_query = None
if "clarification_context" not in st.session_state:
    st.session_state.clarification_context = []

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Bacc2Futur",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
.chat-container {
    display: flex;
    margin-bottom: 1rem;
    align-items: flex-start;
    max-width: 800px;
}

.user-message {
    justify-content: flex-start;
    margin-right: 20%;
}

.assistant-message {
    flex-direction: row-reverse;
    margin-left: 20%;
}

.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 10px;
    font-size: 20px;
}

.user-avatar {
    background-color: #E8F5E9;
    color: #2E7D32;
    border: 2px solid #2E7D32;
}

.assistant-avatar {
    background-color: #E3F2FD;
    color: #1565C0;
    border: 2px solid #1565C0;
}

.message-content {
    padding: 1rem;
    border-radius: 15px;
    max-width: 80%;
}

.user-content {
    background-color: #E8F5E9;
    color: #1a1a1a;
    border: 1px solid #C8E6C9;
}

.assistant-content {
    background-color: #E3F2FD;
    color: #1a1a1a;
    border: 1px solid #BBDEFB;
}

.stTextInput>div>div>input {
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

# Interface utilisateur
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("<h1 style='text-align: center;'>📖 Bacc2Futur - Assistant d'orientation académique</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Bienvenue sur Bacc2Futur, un assistant pour vous aider à trouver la formation et l'orientation qui vous convient.</p>", unsafe_allow_html=True)

# Affichage des messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="chat-container user-message">
                <div class="avatar user-avatar">👤</div>
                <div class="message-content user-content">
                    {message["content"]}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-container assistant-message">
                <div class="avatar assistant-avatar">🤖</div>
                <div class="message-content assistant-content">
                    {message["content"]}
                </div>
            </div>
        """, unsafe_allow_html=True)

# Zone de saisie
if prompt := st.chat_input("Posez votre question ici"):
    # Message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
        <div class="chat-container user-message">
            <div class="avatar user-avatar">👤</div>
            <div class="message-content user-content">
                {prompt}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Réponse de l'assistant
    response_data = conversation_chat(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response_data["response"]})
    st.markdown(f"""
        <div class="chat-container assistant-message">
            <div class="avatar assistant-avatar">🤖</div>
            <div class="message-content assistant-content">
                {response_data["response"]}
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.session_state.conversation_history.extend([prompt, response_data["response"]])
