import os
import re
import streamlit as st
import pandas as pd
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import chromadb
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from rank_bm25 import BM25Okapi
from langchain.schema import BaseRetriever, Document
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator, PrivateAttr
import traceback 

# --- Configuration Sécurisée de la Clé API ---
try:
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyBLhTbSnCpyPFl_SeHI2zd9gRrKa9M1Ssg'
except KeyError:
    if "GOOGLE_API_KEY" not in os.environ:
        print("Clé API Google (GOOGLE_API_KEY) non trouvée. Veuillez la configurer dans les secrets Streamlit ou les variables d'environnement.")
        
    
# Initialisation de ChromaDB
CHROMA_DB_PATH = "./chroma_db"
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection("rag_chatbot")
except Exception as e:
    print(f"Erreur lors de l'initialisation de ChromaDB à {CHROMA_DB_PATH}: {e}")
    
    
# chargement et indexation des données
@st.cache_resource(show_spinner="Chargement et indexation des documents...")
def load_and_index_pdfs(directory, persist_directory):
    try:
        loader = DirectoryLoader(directory, loader_cls=PyPDFLoader, show_progress=True)
        documents = loader.load()
        if not documents:
            st.warning(f"Aucun document PDF trouvé dans le dossier : {directory}")
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        print(f"{len(texts)} chunks indexés avec succès.")
        return db
    except Exception as e:
        st.error(f"Erreur lors du chargement/indexation des PDFs: {e}")
        return None

# Chargement des documents ou de la base existante

pdf_directory = "data/pdfs"
vector_db = None

# Vérifier si la base de données existe déjà
if os.path.exists(CHROMA_DB_PATH):
    try:
        print(f"Chargement de la base vectorielle depuis {CHROMA_DB_PATH}...")
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH,
                           embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        print("Base vectorielle chargée.")
    except Exception as e:
        st.warning(f"Erreur lors du chargement de la base existante: {e}. Tentative de réindexation...")
        vector_db = None 

# Si la base n'a pas été chargée (ou erreur), tenter de la créer/réindexer
if vector_db is None:
    if os.path.exists(pdf_directory) and os.listdir(pdf_directory):
        vector_db = load_and_index_pdfs(pdf_directory, CHROMA_DB_PATH)
    else:
        st.warning(f"Le dossier PDF '{pdf_directory}' est vide ou n'existe pas. Aucune donnée à indexer.")
        # Initialiser une base vectorielle vide
        try:
             vector_db = Chroma(persist_directory=CHROMA_DB_PATH,
                                embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
             st.info("Base vectorielle vide initialisée.")
        except Exception as e:
             st.error(f"Impossible d'initialiser une base vectorielle vide: {e}")
             st.stop()


# --- Vérification si vector_db a été correctement initialisé ---
if vector_db is None:
    st.error("Impossible de charger ou créer la base de données vectorielle. L'application ne peut pas continuer.")
    st.stop()


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
1. Comprendre précisément la demande et poser des questions de clarification si nécessaire.
2. Fournir une réponse claire, détaillée et adaptée au niveau et au contexte de l’utilisateur.
3. S’adapter au ton de l’utilisateur.
4. Structurer tes réponses pour une meilleure lisibilité (ex: listes à puces).
5. Ne pas inventer d’informations. Si tu ne sais pas, dis-le clairement.
6. Utiliser le contexte fourni pour répondre.

Exemples de demandes
• "Peux-tu m'expliquer comment fonctionne Parcoursup ?"
• "Comment structurer une lettre de motivation efficace ?"
• "Quelle est la rémunération d’un apprenti en France ?"
• "propose moi des formations après le baccalauréat "


Limites et restrictions
1. Pas de conseils médicaux ou juridiques.
2. Pas d’événements récents au-delà de la date de mise à jour des données.
3. Pas d’opinions personnelles ou d’émotions.

Question: {question}""", 
)


# Création d'un retriever personnalisé avec BM25
class BM25EnhancedRetriever(BaseRetriever):
    _vectorstore: Chroma = PrivateAttr()
    _k: int = PrivateAttr()
    _base_retriever: BaseRetriever = PrivateAttr()
    _bm25: BM25Okapi = PrivateAttr(default=None)
    _corpus: List[str] = PrivateAttr(default=None)
    _docs_map: List[Document] = PrivateAttr(default=None)

    def __init__(self, vectorstore: Chroma, k: int = 4):
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k
    
        # --- Augmenter k pour le retriever de base pour avoir plus de candidats pour BM25 ---
        self._base_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 5})

    def _initialize_bm25(self, docs: List[Document]):
        """Initialise ou met à jour l'index BM25."""
        self._docs_map = docs
        self._corpus = [doc.page_content for doc in docs]
        tokenized_corpus = [doc.split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Récupérer un plus grand nombre de documents initiaux
        initial_docs = self._base_retriever.get_relevant_documents(query)
        if not initial_docs:
            return []

        self._initialize_bm25(initial_docs)

        # Tokeniser la requête
        tokenized_query = query.split()

        # Calculer les scores BM25
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Combiner documents et scores
        scored_docs = list(zip(self._docs_map, bm25_scores))

        # Trier par score BM25 (décroissant)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Retourner les k meilleurs documents uniques
        final_docs = []
        seen_content = set()
        for doc, score in scored_docs:
            if len(final_docs) >= self._k:
                break
            # --- Éviter les doublons basés sur le contenu ---
            if doc.page_content not in seen_content:
                final_docs.append(doc)
                seen_content.add(doc.page_content)


        return final_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# --- Initialisation du LLM avec gestion d'erreur ---
try:
    # LLM pour générer des requêtes alternatives pour le retriever
    llm_for_retriever = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=500)

    # LLM principal pour la génération de réponse
    main_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=500)

    # LLM pour les fonctions utilitaires (clarification, etc.)
    llm_utils = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5, max_tokens=500) 

except Exception as e:
    st.error(f"Erreur lors de l'initialisation des modèles Google Generative AI: {e}")
    st.stop()


# --- Initialisation du Retriever avec gestion d'erreur ---
try:
    retriever = MultiQueryRetriever.from_llm(
        retriever=BM25EnhancedRetriever(vector_db, k=4), 
        llm=llm_for_retriever,
        prompt=QUERY_PROMPT 
    )
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du MultiQueryRetriever: {e}")
    st.stop()


# RAG prompt template
template = """Répond à la question de l'utilisateur en te basant UNIQUEMENT sur le contexte suivant. Si le contexte ne contient pas la réponse, dis que tu ne sais pas. Ne fais pas référence au contexte dans ta réponse finale.

Contexte:
{context}

Question: {query}

Réponse:
""" 

prompt = ChatPromptTemplate.from_template(template)


# Création de la chaîne RAG
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | main_llm 
    | StrOutputParser()
)

# --- Fonctions utilitaires ---

# Fonction pour détecter si une question de clarification est nécessaire
def needs_clarification(query):
    clarification_prompt = ChatPromptTemplate.from_template("""
    Analyse la question suivante de l'utilisateur. L'utilisateur cherche des informations sur l'orientation scolaire et professionnelle en France.
    La question est-elle suffisamment précise pour y répondre directement ou est-elle trop vague/manque de contexte (par exemple, l'utilisateur demande "des formations" sans préciser de domaine, de niveau, etc.)?

    Question: {query}

    Réponds par "True" si la question est vague ou nécessite plus de contexte pour une réponse utile.
    Réponds par "False" si la question est claire et spécifique.
    Réponse (True/False uniquement):
    """)
    try:
        clarif_chain = clarification_prompt | llm_utils | StrOutputParser()
        response = clarif_chain.invoke({"query": query})
        # --- Vérification plus robuste de la réponse ---
        return response.strip().lower() == "true"
    except Exception as e:
        st.warning(f"Erreur dans needs_clarification: {e}")
        return False 

# Fonction pour générer des questions de clarification
def generate_clarifying_questions(query):
    question_prompt = ChatPromptTemplate.from_template("""
    En tant qu'assistant d'orientation, génère 2 ou 3 questions pertinentes et concises pour aider l'utilisateur à préciser sa demande suivante. Les questions doivent viser à obtenir des informations manquantes (domaine d'intérêt, niveau d'études, préférences, etc.).

    Demande de l'utilisateur:
    {query}

    Format: Retourne uniquement les questions, une par ligne, commençant par un tiret (-). Ne numérote pas les questions.
    Exemple:
    - Quel domaine d'études vous intéresse ?
    - Quel est votre niveau d'études actuel ?
    """)
    try:
        gen_q_chain = question_prompt | llm_utils | StrOutputParser()
        response = gen_q_chain.invoke({"query": query})
        # --- Nettoyage pour s'assurer que ce sont bien des questions ---
        questions = [q.strip() for q in response.strip().split('\n') if q.strip().startswith('-')]
        return questions if questions else ["Pourriez-vous préciser votre demande ?"] # Fallback
    except Exception as e:
        st.warning(f"Erreur dans generate_clarifying_questions: {e}")
        return ["Pourriez-vous préciser votre demande ?"] # Fallback

# Fonction pour nettoyer la réponse
def clean_response(response):
    if not response or response.strip() == "":
        return "Je n'ai pas trouvé de réponse pertinente à votre question dans les documents disponibles. Pourriez-vous la reformuler ou me donner plus de détails ?"
    cleaned = re.sub(r"```json\s*(.*?)\s*```", r"\1", response, flags=re.DOTALL)
    cleaned = re.sub(r"Je ne dispose pas d'informations à ce sujet.*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"Je ne peux pas répondre à cette question.*", "", cleaned, flags=re.IGNORECASE).strip()
    
    if not cleaned:
         return "Je n'ai pas trouvé de réponse pertinente à votre question dans les documents disponibles. Pourriez-vous la reformuler ou me donner plus de détails ?"
    return cleaned.strip()

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
        cleaned_word = re.sub(r'[^\w\s]', '', word) 
        if cleaned_word in greetings:
            return greetings[cleaned_word]
    return None



# Fonction principale de conversation
def conversation_chat(query):
    try:
        # Vérifier les salutations simples
        greeting = detect_greeting(query)
        if greeting and len(query.split()) <= 2:
            return {
                "response": f"{greeting}, comment puis-je vous aider aujourd'hui ?",
                "needs_clarification": False
            }

        # Gérer les réponses aux questions de clarification
        if st.session_state.get('awaiting_clarification', False):
            # Ajouter la réponse de l'utilisateur au contexte de clarification
            st.session_state.clarification_context.append(query)
            # Construire une requête combinée plus naturelle
            combined_query = f"Ma question initiale était : '{st.session_state.original_query}'. Voici mes précisions : {' '.join(st.session_state.clarification_context)}"
            st.info("Merci pour ces précisions. Recherche en cours...")

            # Utiliser la chaîne RAG principale avec la requête combinée
            with st.spinner("Réflexion avec les précisions..."):
                final_response = chain.invoke(combined_query) # Utilise la chaîne RAG

            # Réinitialiser l'état d'attente
            st.session_state.awaiting_clarification = False
            st.session_state.original_query = None
            st.session_state.clarification_context = []

            return {
                "response": clean_response(final_response),
                "needs_clarification": False
            }

        # Vérifier si la nouvelle question nécessite clarification AVANT d'appeler la chaîne RAG
        if needs_clarification(query):
            st.session_state.awaiting_clarification = True
            st.session_state.original_query = query
            st.session_state.clarification_context = [] # Réinitialiser le contexte
            clarifying_questions = generate_clarifying_questions(query)
            # --- Formatage amélioré des questions ---
            response_text = "Pour mieux vous aider, pourriez-vous préciser les points suivants ?\n\n" + "\n".join(clarifying_questions)
            return {
                "response": response_text,
                "needs_clarification": True
            }

        # Question directe (utiliser la chaîne RAG principale)
        with st.spinner("Recherche d'informations et génération de la réponse..."):
            response_content = chain.invoke(query) # Utilise la chaîne RAG

        final_response_text = clean_response(response_content)

        # Ajouter la salutation si détectée dans une question plus longue
        if greeting and not final_response_text.lower().startswith(greeting.lower()):
             final_response_text = f"{greeting} ! {final_response_text}"

        return {
            "response": final_response_text,
            "needs_clarification": False
        }

    except Exception as e:
        # --- AJOUT POUR LE DEBUG ---
        print(f"ERREUR DÉTAILLÉE dans conversation_chat: {e}")
        import traceback
        traceback.print_exc() # Affiche la trace complète de l'erreur
        # --- FIN DE L'AJOUT ---
        st.error(f"Une erreur est survenue lors du traitement de votre demande: {str(e)}")
        return {
            "response": "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer.",
            "needs_clarification": False
        }


# Initialisation des variables de session 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "original_query" not in st.session_state:
    st.session_state.original_query = None
if "clarification_context" not in st.session_state:
    st.session_state.clarification_context = []
    
    
    
    
    

# Configuration de la page Streamlit 
st.set_page_config(
    page_title="Bacc2Future", # Correction nom
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés (Correction des > et <)
st.markdown("""
<style>
/* Styles généraux */
body {
    font-family: 'Arial', sans-serif; /* Police plus standard */
}

/* Conteneur principal */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 5rem; /* Espace pour le chat input fixe */
}

/* Titre et description */
h1, .stMarkdown p:first-of-type {
    text-align: center;
    margin-bottom: 1.5rem;
}
h1 {
    color: #1565C0; /* Couleur bleue pour le titre */
}

/* Conteneur de chat */
.chat-container {
    display: flex;
    margin-bottom: 1rem;
    align-items: flex-start;
    max-width: 850px; /* Légèrement plus large */
    margin-left: auto;
    margin-right: auto;
}

/* Alignement des messages */
.user-message {
    justify-content: flex-start;
}
.assistant-message {
    justify-content: flex-start;
}

/* Avatars */
.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 10px; /* Espace entre avatar et message */
    font-size: 20px;
    flex-shrink: 0;
    order: 0; /* Avatar toujours à gauche */
}
.user-avatar {
    background-color: #E8F5E9; /* Vert clair */
    color: #2E7D32; /* Vert foncé */
    border: 1px solid #A5D6A7; /* Bordure verte plus douce */
}
.assistant-avatar {
    background-color: #E3F2FD; /* Bleu clair */
    color: #1565C0; /* Bleu foncé */
    border: 1px solid #90CAF9; /* Bordure bleue plus douce */
}

/* Contenu des messages */
.message-content {
    padding: 0.8rem 1.2rem; /* Padding ajusté */
    border-radius: 18px; /* Coins plus arrondis */
    max-width: calc(100% - 50px); /* Largeur max ajustée */
    word-wrap: break-word;
    order: 1; /* Contenu à droite de l'avatar */
    box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Ombre légère */
}
.user-content {
    background-color: #FFFFFF; /* Fond blanc pour l'utilisateur */
    color: #333;
    border: 1px solid #e0e0e0;
    /* align-self: flex-end; /* Pourrait être utilisé pour aligner à droite si souhaité */
}
.assistant-content {
    background-color: #F1F8FF; /* Fond bleu très clair pour l'assistant */
    color: #333;
    border: 1px solid #d1e6ff;
}

/* Champ de saisie */
.stTextInput > div > div > input {
    border-radius: 20px;
    border: 1px solid #ccc;
    padding: 10px 15px;
}
/* Style pour le conteneur du chat_input pour le fixer en bas (optionnel) */
/* div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: white;
    padding: 10px 0;
    z-index: 1000;
    border-top: 1px solid #eee;
} */

/* Style des onglets */
div[data-baseweb="tab-list"] {
    justify-content: center; /* Centrer les onglets */
    margin-bottom: 1.5rem;
}
button[data-baseweb="tab"] {
    font-size: 1rem;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📖 Bacc2Future - Assistant d'orientation académique</h1>", unsafe_allow_html=True)
st.markdown("<p>Bienvenue sur Bacc2Future, votre assistant pour l'orientation académique et professionnelle.</p>", unsafe_allow_html=True)

# Création des onglets
tab1, tab2 = st.tabs(["💬 Assistant IA", "📋 Suivi "])

with tab1:
    st.markdown("### Posez votre question à l'assistant")

    chat_display_container = st.container()
    with chat_display_container:
        # Affichage des messages existants
        for message in st.session_state.messages:
            avatar_icon = "👤" if message["role"] == "user" else "🤖"
            avatar_class = "user-avatar" if message["role"] == "user" else "assistant-avatar"
            content_class = "user-content" if message["role"] == "user" else "assistant-content"
            container_class = "user-message" if message["role"] == "user" else "assistant-message"

            st.markdown(f"""
                <div class="chat-container {container_class}">
                    <div class="avatar {avatar_class}">{avatar_icon}</div>
                    <div class="message-content {content_class}">
                        {message["content"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response_data = conversation_chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response_data["response"]})
        st.rerun()



with tab2:
        
    # Initialisation de l'état de session pour stocker les candidatures
    if "candidatures" not in st.session_state:
        st.session_state.candidatures = []
    if "candidature_a_modifier" not in st.session_state:
        st.session_state.candidature_a_modifier = None

    # Fonction pour supprimer une candidature
    def supprimer_candidature(index):
        st.session_state.candidatures.pop(index)
        st.rerun()

    # Fonction pour définir la candidature à modifier
    def set_candidature_a_modifier(index):
        st.session_state.candidature_a_modifier = index
        st.rerun()

    # Fonction pour modifier une candidature existante
    def modifier_candidature(index, entreprise, poste, type_candidature, statut, date):
        st.session_state.candidatures[index].update({
            "Entreprise": entreprise,
            "Poste": poste,
            "Type": type_candidature,
            "Statut": statut,
            "Date": date.strftime("%Y-%m-%d")
        })
        st.session_state.candidature_a_modifier = None
        st.rerun()

    # Fonction pour afficher le formulaire de modification
    def afficher_formulaire_modification(index, candidature):
        with st.form(f"modification_candidature_{index}"):
            st.markdown("### ✏️ Modifier la Candidature")
            entreprise = st.text_input("Entreprise / Établissement", value=candidature["Entreprise"])
            poste = st.text_input("Poste / Formation", value=candidature["Poste"])
            type_candidature = st.selectbox(
                "Type",
                ["Stage", "Alternance", "Emploi", "Formation"],
                index=["Stage", "Alternance", "Emploi", "Formation"].index(candidature["Type"])
            )
            statut = st.selectbox(
                "Statut",
                ["En cours", "Accepté", "Refusé"],
                index=["En cours", "Accepté", "Refusé"].index(candidature["Statut"])
            )
            date = st.date_input("Date de candidature", datetime.strptime(candidature["Date"], "%Y-%m-%d"))
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 Enregistrer"):
                    if entreprise and poste:
                        modifier_candidature(index, entreprise, poste, type_candidature, statut, date)
            with col2:
                if st.form_submit_button("❌ Annuler"):
                    st.session_state.candidature_a_modifier = None
                    st.rerun()

    # Fonction pour afficher le tableau de bord
    def afficher_tableau():
        st.markdown("## 📋 Suivi des Candidatures")
        
        if not st.session_state.candidatures:
            st.info("Aucune candidature enregistrée pour le moment.")
            return
        
        df = pd.DataFrame(st.session_state.candidatures)
        
        # Filtres dynamiques
        col1, col2 = st.columns(2)
        with col1:
            statut_filter = st.selectbox("Filtrer par statut", ["Tous", "En cours", "Accepté", "Refusé"])
        with col2:
            type_filter = st.selectbox("Filtrer par type", ["Tous", "Stage", "Alternance", "Emploi", "Formation"])
        
        filtered_df = df.copy()
        if statut_filter != "Tous":
            filtered_df = filtered_df[filtered_df["Statut"] == statut_filter]
        if type_filter != "Tous":
            filtered_df = filtered_df[filtered_df["Type"] == type_filter]
        
        # Afficher le tableau avec les boutons de modification
        st.markdown("### Liste des candidatures")
        for index, row in df.iterrows():
            with st.expander(f"{row['Entreprise']} - {row['Poste']} ({row['Statut']})"):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**Type:** {row['Type']}")
                    st.write(f"**Date:** {row['Date']}")
                    st.write(f"**Statut:** {row['Statut']}")
                with col2:
                    if st.button("✏️ Modifier", key=f"edit_{index}"):
                        set_candidature_a_modifier(index)
                with col3:
                    if st.button("🗑️ Supprimer", key=f"delete_{index}"):
                        if st.button("Confirmer ❌", key=f"confirm_delete_{index}"):
                            supprimer_candidature(index)
            
            # Afficher le formulaire de modification si cette candidature est sélectionnée
            if st.session_state.candidature_a_modifier == index:
                afficher_formulaire_modification(index, row)

        st.markdown("### Vue d'ensemble")
        st.dataframe(filtered_df, use_container_width=True)

    # Fonction pour ajouter une candidature
    def ajouter_candidature():
        with st.form("ajout_candidature"):
            st.markdown("### ➕ Ajouter une Candidature")
            entreprise = st.text_input("Entreprise / Établissement")
            poste = st.text_input("Poste / Formation")
            type_candidature = st.selectbox("Type", ["Stage", "Alternance", "Emploi", "Formation"])
            statut = st.selectbox("Statut", ["En cours", "Accepté", "Refusé"])
            date = st.date_input("Date de candidature", datetime.today())
            submit = st.form_submit_button("Ajouter")
            
            if submit and entreprise and poste:
                st.session_state.candidatures.append({
                    "Entreprise": entreprise,
                    "Poste": poste,
                    "Type": type_candidature,
                    "Statut": statut,
                    "Date": date.strftime("%Y-%m-%d")
                })
                st.success("Candidature ajoutée avec succès !")
                st.rerun()
            
    # Interface principale
    #st.set_page_config(page_title="Suivi des Candidatures", layout="wide")
    st.title("🎯 Tableau de Suivi des Candidatures")

    ajouter_candidature()
    afficher_tableau()
        

