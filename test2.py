import os
import re
import streamlit as st
import pandas as pd
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# Correction Import Chroma: Utiliser le nouveau package
from langchain_chroma import Chroma
# Correction Import ChromaDB Client: Ajouter l'import si utilisé directement
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
# Correction Import Pydantic: Importer field_validator
from pydantic import BaseModel, Field, field_validator, PrivateAttr
import traceback

# --- Configuration Sécurisée de la Clé API ---
try:
    os.environ["GOOGLE_API_KEY"] = 'APIkey'
except KeyError:
    if "GOOGLE_API_KEY" not in os.environ:
        print("Clé API Google (GOOGLE_API_KEY) non trouvée. Veuillez la configurer dans les secrets Streamlit ou les variables d'environnement.")
        
    

# --- Constantes ---
CHROMA_DB_PATH = "./chroma_db" # Garder le même chemin si c'est celui que tu utilises
PDF_DIRECTORY = "data/pdfs"

# --- Initialisation ChromaDB Client (si nécessaire) ---
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # La collection sera gérée par LangChain Chroma lors de l'initialisation/chargement
    # chroma_collection = chroma_client.get_or_create_collection("rag_chatbot") # Optionnel si géré par LangChain
    print(f"Client ChromaDB initialisé pour le chemin : {CHROMA_DB_PATH}")
except Exception as e:
    # Utiliser st.error pour afficher dans l'interface Streamlit
    st.error(f"Erreur lors de l'initialisation du client ChromaDB à {CHROMA_DB_PATH}: {e}")
    # st.stop() # Peut-être trop strict, dépend si l'app peut fonctionner sans


# --- Modèle Pydantic pour la Structuration ---
class StructuredChunk(BaseModel):
    """Représente un chunk de texte structuré extrait d'un document."""
    text_content: str = Field(..., description="Le contenu textuel réel du chunk")
    source: str = Field(..., description="Chemin ou identifiant du document source")
    page: Optional[int] = Field(None, description="Numéro de page si disponible")
    chunk_sequence: int = Field(..., description="Index séquentiel du chunk dans le document")

    # Correction Pydantic Validator: Utiliser field_validator et classmethod
    @field_validator('text_content')
    @classmethod
    def text_content_must_not_be_empty(cls, v: str) -> str: # Ajout des type hints
        if not v or not v.strip():
            raise ValueError('Le contenu textuel ne doit pas être vide')
        return v

    def generate_id(self) -> str:
        """Génère un ID unique pour le chunk."""
        content_hash = hashlib.md5(self.text_content.encode()).hexdigest()[:8]
        page_str = f"p{self.page}" if self.page is not None else "pNA"
        filename = os.path.basename(self.source)
        return f"{filename}-{page_str}-chunk{self.chunk_sequence}-{content_hash}"

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convertit les champs Pydantic en métadonnées pour Chroma."""
        metadata = {
            "source": self.source,
            "chunk_sequence": self.chunk_sequence,
        }
        if self.page is not None:
            metadata["page"] = self.page

        cleaned_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned_metadata[key] = value
            elif value is None:
                 cleaned_metadata[key] = "N/A"
            else:
                cleaned_metadata[key] = str(value)
        return cleaned_metadata

# --- Fonction de Chargement, Structuration et Indexation ---
# Renommée pour correspondre à l'appel plus bas si nécessaire, ou garder l'ancien nom
@st.cache_resource(show_spinner="Chargement, structuration et indexation des documents...")
def load_structure_and_index_pdfs(directory: str, persist_directory: str) -> Optional[Chroma]:
    """
    Charge les PDFs, les découpe, structure les chunks avec Pydantic,
    et les indexe dans Chroma.
    """
    try:
        if not os.path.isdir(directory):
            st.warning(f"Le dossier spécifié n'existe pas : {directory}")
            return None

        loader = DirectoryLoader(directory, loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True, silent_errors=True)
        raw_docs = loader.load()

        if not raw_docs:
            st.warning(f"Aucun document PDF trouvé ou chargé depuis le dossier : {directory}")
            # Essayer d'initialiser une DB vide si elle n'existe pas
            if not os.path.exists(persist_directory):
                 try:
                     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                     db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                     st.info("Base vectorielle vide initialisée car aucun document trouvé.")
                     return db
                 except Exception as init_e:
                     st.error(f"Impossible d'initialiser une base vectorielle vide: {init_e}")
                     return None
            else:
                return None # DB existe mais pas de nouveaux docs

        print(f"Chargé {len(raw_docs)} pages/objets bruts.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(raw_docs)
        print(f"Découpé en {len(split_docs)} chunks de texte.")

        structured_chunks: List[StructuredChunk] = []
        doc_chunk_counters: Dict[str, int] = {}

        print("Structuration des chunks de texte avec Pydantic...")
        for i, doc_chunk in enumerate(split_docs):
            source = doc_chunk.metadata.get("source", "source_inconnue")
            page = doc_chunk.metadata.get("page", None)

            current_chunk_seq = doc_chunk_counters.get(source, 0)
            doc_chunk_counters[source] = current_chunk_seq + 1

            try:
                chunk_obj = StructuredChunk(
                    text_content=doc_chunk.page_content,
                    source=source,
                    page=page,
                    chunk_sequence=current_chunk_seq
                )
                structured_chunks.append(chunk_obj)
            except Exception as pydantic_error:
                st.warning(f"Chunk {i} ignoré ('{source}', page {page}) : Erreur de structuration/validation - {pydantic_error}")
                continue

        if not structured_chunks:
             st.warning("Aucun chunk n'a pu être structuré après validation.")
             return None # Pas de nouveaux chunks à ajouter

        chunk_texts = [chunk.text_content for chunk in structured_chunks]
        chunk_metadatas = [chunk.to_chroma_metadata() for chunk in structured_chunks]
        chunk_ids = [chunk.generate_id() for chunk in structured_chunks]

        print(f"Indexation de {len(chunk_texts)} chunks structurés...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        db = Chroma.from_texts(
            texts=chunk_texts,
            embedding=embeddings,
            metadatas=chunk_metadatas,
            ids=chunk_ids,
            persist_directory=persist_directory
        )
        db.persist() # Forcer la persistance
        print(f"{len(chunk_texts)} chunks structurés et indexés avec succès dans {persist_directory}.")
        st.success(f"{len(chunk_texts)} chunks structurés et indexés.")
        return db

    except Exception as e:
        st.error(f"Erreur majeure lors du processus de chargement/structuration/indexation: {e}")
        traceback.print_exc()
        return None

# --- Chargement/Initialisation de la Base Vectorielle ---
vector_db: Optional[Chroma] = None
embeddings_for_loading = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 1. Essayer de charger la base existante
if os.path.exists(CHROMA_DB_PATH):
    try:
        print(f"Chargement de la base vectorielle depuis {CHROMA_DB_PATH}...")
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH,
                           embedding_function=embeddings_for_loading)
        print("Base vectorielle chargée.")
    except Exception as e:
        st.warning(f"Erreur lors du chargement de la base existante: {e}. Tentative de réindexation...")
        vector_db = None

# 2. Si non chargée, tenter de créer/réindexer depuis les PDFs
if vector_db is None:
    if os.path.exists(PDF_DIRECTORY) and os.listdir(PDF_DIRECTORY):
        print(f"Tentative de création/indexation depuis le dossier {PDF_DIRECTORY}...")
        # Correction Nom Fonction: Appeler la fonction définie ci-dessus
        vector_db = load_structure_and_index_pdfs(PDF_DIRECTORY, CHROMA_DB_PATH)
    else:
        st.warning(f"Le dossier PDF '{PDF_DIRECTORY}' est vide ou n'existe pas. Aucune donnée à indexer.")
        # Essayer d'initialiser une base vide si elle n'existe pas encore
        if not os.path.exists(CHROMA_DB_PATH):
            try:
                 vector_db = Chroma(persist_directory=CHROMA_DB_PATH,
                                    embedding_function=embeddings_for_loading)
                 st.info("Base vectorielle vide initialisée.")
            except Exception as e_init:
                 st.error(f"Impossible d'initialiser une base vectorielle vide: {e_init}")
        else:
             st.warning("Impossible de réindexer (pas de PDFs) et la base existante n'a pas pu être chargée correctement.")


# --- Vérification Finale et Activation RAG ---
rag_enabled = False
if vector_db is not None:
    rag_enabled = True
else:
    st.error("Impossible de charger ou créer la base de données vectorielle. Le Chatbot IA sera désactivé.")


# --- Définition du Prompt RAG ---
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Rôle & Mission
Tu es un assistant virtuel spécialisé dans l’orientation académique et professionnelle des élèves et étudiants en France...
[... Reste du prompt inchangé ...]
Question: {question}""",
)

# --- Retriever Personnalisé BM25 ---
class BM25EnhancedRetriever(BaseRetriever):
    _vectorstore: Chroma = PrivateAttr()
    _k: int = PrivateAttr()
    _base_retriever: BaseRetriever = PrivateAttr()
    _bm25: Optional[BM25Okapi] = PrivateAttr(default=None)
    _docs_map: List[Document] = PrivateAttr(default_factory=list)
    _corpus: List[str] = PrivateAttr(default_factory=list)

    def __init__(self, vectorstore: Chroma, k: int = 4):
        super().__init__()
        self._vectorstore = vectorstore
        self._k = k
        self._base_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 5})

    def _initialize_bm25(self, docs: List[Document]):
        """Initialise ou met à jour l'index BM25."""
        self._docs_map = docs
        self._corpus = [doc.page_content for doc in docs]
        if self._corpus:
            tokenized_corpus = [doc.split() for doc in self._corpus]
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None

    # Correction LangChain Deprecation: Utiliser invoke au lieu de get_relevant_documents si possible
    # Garder _get_relevant_documents pour la compatibilité interne de la classe pour l'instant
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Utiliser invoke sur le retriever de base
        try:
            initial_docs = self._base_retriever.invoke(query)
        except AttributeError: # Fallback pour anciennes versions
             initial_docs = self._base_retriever.get_relevant_documents(query)

        if not initial_docs:
            return []

        self._initialize_bm25(initial_docs)

        if not self._bm25:
             return initial_docs[:self._k]

        tokenized_query = query.split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        scored_docs = list(zip(self._docs_map, bm25_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        final_docs = []
        seen_content = set()
        for doc, score in scored_docs:
            if len(final_docs) >= self._k:
                break
            if doc.page_content not in seen_content:
                final_docs.append(doc)
                seen_content.add(doc.page_content)

        if not final_docs and initial_docs:
            return initial_docs[:self._k]

        return final_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Utiliser ainvoke sur le retriever de base
        try:
            initial_docs = await self._base_retriever.ainvoke(query)
        except AttributeError: # Fallback
            initial_docs = await self._base_retriever.aget_relevant_documents(query)

        if not initial_docs: return []
        # Le reste est synchrone
        self._initialize_bm25(initial_docs)
        if not self._bm25: return initial_docs[:self._k]
        tokenized_query = query.split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        scored_docs = list(zip(self._docs_map, bm25_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = []
        seen_content = set()
        for doc, score in scored_docs:
             if len(final_docs) >= self._k: break
             if doc.page_content not in seen_content:
                 final_docs.append(doc); seen_content.add(doc.page_content)
        if not final_docs and initial_docs: return initial_docs[:self._k]
        return final_docs


# --- Initialisation LLMs et Chaîne RAG (Seulement si RAG activé) ---
chain = None
llm_utils = None

if rag_enabled:
    try:
        llm_for_retriever = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=500)
        main_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, max_tokens=500)
        llm_utils = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5, max_tokens=500)

        retriever = MultiQueryRetriever.from_llm(
            retriever=BM25EnhancedRetriever(vector_db, k=4),
            llm=llm_for_retriever,
            prompt=QUERY_PROMPT
        )

        rag_template = """Répond à la question de l'utilisateur en te basant UNIQUEMENT sur le contexte suivant. Si le contexte ne contient pas la réponse, dis que tu ne sais pas. Ne fais pas référence au contexte dans ta réponse finale.

        Contexte:
        {context}

        Question: {query} # Attend 'query'

        Réponse:
        """
        rag_prompt = ChatPromptTemplate.from_template(rag_template)

        # Correction Clé Chaîne RAG: Utiliser "query" au lieu de "question"
        chain = (
            {"context": retriever, "query": RunnablePassthrough()} # Fournit 'query'
            | rag_prompt
            | main_llm
            | StrOutputParser()
        )
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation des modèles ou de la chaîne RAG: {e}")
        rag_enabled = False # Désactiver RAG si l'initialisation échoue

# --- Fonctions Utilitaires (Clarification, Nettoyage, Salutation) ---
# (Avec vérification si llm_utils est initialisé)

def needs_clarification(query):
    if not llm_utils: return False
    clarification_prompt = ChatPromptTemplate.from_template("""
    Analyse la question suivante... Est-elle vague ? ...
    Question: {query}
    Réponds par "True" ou "False".
    Réponse (True/False uniquement):
    """)
    try:
        clarif_chain = clarification_prompt | llm_utils | StrOutputParser()
        response = clarif_chain.invoke({"query": query})
        return response.strip().lower() == "true"
    except Exception as e:
        st.warning(f"Erreur dans needs_clarification: {e}")
        # Logguer l'erreur pour le debug
        print(f"--- Erreur needs_clarification détaillée ---")
        traceback.print_exc()
        print(f"--- Fin Erreur needs_clarification ---")
        return False

def generate_clarifying_questions(query):
    if not llm_utils: return ["Pourriez-vous préciser votre demande ?"]
    question_prompt = ChatPromptTemplate.from_template("""
    Génère 2-3 questions pertinentes pour préciser la demande suivante...
    Demande: {query}
    Format: Retourne uniquement les questions, une par ligne, commençant par un tiret (-).
    """)
    try:
        gen_q_chain = question_prompt | llm_utils | StrOutputParser()
        response = gen_q_chain.invoke({"query": query})
        questions = [q.strip() for q in response.strip().split('\n') if q.strip().startswith('-')]
        return questions if questions else ["Pourriez-vous préciser votre demande ?"]
    except Exception as e:
        st.warning(f"Erreur dans generate_clarifying_questions: {e}")
        # Logguer l'erreur pour le debug
        print(f"--- Erreur generate_clarifying_questions détaillée ---")
        traceback.print_exc()
        print(f"--- Fin Erreur generate_clarifying_questions ---")
        return ["Pourriez-vous préciser votre demande ?"]

def clean_response(response):
    if not response or not response.strip():
        return "Je n'ai pas trouvé de réponse pertinente. Pourriez-vous reformuler ?"
    cleaned = re.sub(r"```json\s*(.*?)\s*```", r"\1", response, flags=re.DOTALL)
    cleaned = re.sub(r"Je ne dispose pas d'informations à ce sujet.*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"Je ne peux pas répondre à cette question.*", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
         return "Je n'ai pas trouvé de réponse pertinente. Pourriez-vous reformuler ?"
    return cleaned.strip()

def detect_greeting(text):
    greetings = {"bonjour": "Bonjour", "bonsoir": "Bonsoir", "salut": "Salut", "hey": "Hey", "coucou": "Coucou", "hello": "Hello"}
    first_words = text.lower().split()[:2]
    for word in first_words:
        cleaned_word = re.sub(r'[^\w\s]', '', word)
        if cleaned_word in greetings:
            return greetings[cleaned_word]
    return None

# --- Fonction Principale de Conversation (avec gestion d'erreurs fines) ---
def conversation_chat(query):
    if not rag_enabled or not chain:
        return {
            "response": "Désolé, le service de questions-réponses est actuellement indisponible.",
            "needs_clarification": False
        }

    try:
        greeting = detect_greeting(query)
        if greeting and len(query.split()) <= 2:
            return {"response": f"{greeting}, comment puis-je vous aider ?", "needs_clarification": False}

        if st.session_state.get('awaiting_clarification', False):
            st.session_state.clarification_context.append(query)
            combined_query = f"Question initiale : '{st.session_state.original_query}'. Précisions : {' '.join(st.session_state.clarification_context)}"
            st.info("Merci. Recherche en cours avec les précisions...")
            final_response = "Désolé, une erreur est survenue lors de la recherche avec vos précisions." # Fallback
            try:
                with st.spinner("Réflexion..."):
                    # Correction Clé Chaîne RAG: Utiliser "query" pour l'invoke si la chaîne l'attend
                    # Ici, la chaîne attend un seul argument (la question), donc on passe combined_query directement
                    final_response = chain.invoke(combined_query)
            except Exception as chain_error:
                st.warning(f"Erreur spécifique lors de l'appel RAG après clarification: {chain_error}")
                print(f"--- Erreur RAG (clarification) détaillée ---")
                traceback.print_exc()
                print(f"--- Fin Erreur RAG (clarification) ---")
                # Gérer l'erreur de quota spécifiquement
                if "ResourceExhausted" in str(chain_error):
                     final_response = "J'ai atteint ma limite de requêtes pour le moment. Veuillez réessayer dans quelques instants."
                # La réponse fallback sera utilisée sinon

            st.session_state.awaiting_clarification = False
            st.session_state.original_query = None
            st.session_state.clarification_context = []
            return {"response": clean_response(final_response), "needs_clarification": False}

        needs_clarif = False
        try:
            needs_clarif = needs_clarification(query)
        except Exception as clarif_error:
             st.warning(f"Erreur lors de la vérification de la clarification: {clarif_error}")
             print(f"--- Erreur needs_clarification détaillée ---")
             traceback.print_exc()
             print(f"--- Fin Erreur needs_clarification ---")
             needs_clarif = False

        if needs_clarif:
            st.session_state.awaiting_clarification = True
            st.session_state.original_query = query
            st.session_state.clarification_context = []
            clarifying_questions = ["Pourriez-vous préciser votre demande ?"] # Fallback
            try:
                clarifying_questions = generate_clarifying_questions(query)
            except Exception as gen_q_error:
                 st.warning(f"Erreur lors de la génération des questions de clarification: {gen_q_error}")
                 print(f"--- Erreur generate_clarifying_questions détaillée ---")
                 traceback.print_exc()
                 print(f"--- Fin Erreur generate_clarifying_questions ---")

            response_text = "Pour mieux vous aider, pourriez-vous préciser ?\n\n" + "\n".join(clarifying_questions)
            return {"response": response_text, "needs_clarification": True}

        response_content = "Désolé, je n'ai pas pu traiter votre demande pour le moment." # Fallback
        try:
            with st.spinner("Recherche et génération de la réponse..."):
                 # Correction Clé Chaîne RAG: Utiliser "query" pour l'invoke si la chaîne l'attend
                 # Ici, la chaîne attend un seul argument (la question), donc on passe query directement
                response_content = chain.invoke(query)
        except Exception as chain_error:
            st.warning(f"Erreur spécifique lors de l'appel RAG direct: {chain_error}")
            print(f"--- Erreur RAG (direct) détaillée ---")
            traceback.print_exc()
            print(f"--- Fin Erreur RAG (direct) ---")
            # Gérer l'erreur de quota spécifiquement
            if "ResourceExhausted" in str(chain_error):
                 response_content = "J'ai atteint ma limite de requêtes pour le moment. Veuillez réessayer dans quelques instants."
            # La réponse fallback sera utilisée sinon

        final_response_text = clean_response(response_content)
        if greeting and not final_response_text.lower().startswith(greeting.lower()):
             final_response_text = f"{greeting} ! {final_response_text}"
        return {"response": final_response_text, "needs_clarification": False}

    except Exception as e:
        print(f"ERREUR GÉNÉRALE INATTENDUE dans conversation_chat: {e}")
        traceback.print_exc()
        st.error(f"Une erreur technique inattendue est survenue: {str(e)}")
        return {"response": "Désolé, une erreur technique majeure m'empêche de répondre.", "needs_clarification": False}


# --- Initialisation Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "awaiting_clarification" not in st.session_state: st.session_state.awaiting_clarification = False
if "original_query" not in st.session_state: st.session_state.original_query = None
if "clarification_context" not in st.session_state: st.session_state.clarification_context = []
if "candidatures" not in st.session_state: st.session_state.candidatures = []
if "candidature_a_modifier" not in st.session_state: st.session_state.candidature_a_modifier = None
if "confirming_delete_index" not in st.session_state: st.session_state.confirming_delete_index = None


# --- Configuration Page Streamlit ---
st.set_page_config(
    page_title="Bacc2Future",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styles CSS ---
st.markdown("""
<style>
/* ... [Styles CSS inchangés] ... */
/* Styles généraux */
body { font-family: 'Arial', sans-serif; }
.main .block-container { padding-top: 2rem; padding-bottom: 5rem; }
h1, .stMarkdown p:first-of-type { text-align: center; margin-bottom: 1.5rem; }
h1 { color: #1565C0; }
.chat-container { display: flex; margin-bottom: 1rem; align-items: flex-start; max-width: 850px; margin-left: auto; margin-right: auto; }
.user-message { justify-content: flex-start; }
.assistant-message { justify-content: flex-start; }
.avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-size: 20px; flex-shrink: 0; order: 0; }
.user-avatar { background-color: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
.assistant-avatar { background-color: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; }
.message-content { padding: 0.8rem 1.2rem; border-radius: 18px; max-width: calc(100% - 50px); word-wrap: break-word; order: 1; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.user-content { background-color: #FFFFFF; color: #333; border: 1px solid #e0e0e0; }
.assistant-content { background-color: #F1F8FF; color: #333; border: 1px solid #d1e6ff; }
.stTextInput > div > div > input { border-radius: 20px; border: 1px solid #ccc; padding: 10px 15px; }
div[data-baseweb="tab-list"] { justify-content: center; margin-bottom: 1.5rem; }
button[data-baseweb="tab"] { font-size: 1rem; padding: 10px 20px; }
.delete-confirmation button { background-color: #f44336; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer; margin-left: 10px;}
.delete-confirmation button:hover { background-color: #d32f2f; }
</style>
""", unsafe_allow_html=True)

# --- Titre et Description ---
st.markdown("<h1>📖 Bacc2Future - Assistant d'orientation académique</h1>", unsafe_allow_html=True)
st.markdown("<p>Bienvenue sur Bacc2Future, votre assistant pour l'orientation académique et professionnelle.</p>", unsafe_allow_html=True)

# --- Onglets ---
tab1, tab2 = st.tabs(["💬 Assistant IA", "📋 Suivi Candidatures"])

# --- Onglet Assistant IA ---
with tab1:
    st.markdown("### Posez votre question à l'assistant")

    chat_display_container = st.container()
    with chat_display_container:
        # Affichage messages existants
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

    # Input utilisateur
    if prompt_input := st.chat_input("Posez votre question ici..." if rag_enabled else "Le Chatbot est désactivé.", disabled=not rag_enabled):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        st.rerun() # Afficher immédiatement le message utilisateur

    # Générer réponse si le dernier message est de l'utilisateur
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
         user_prompt = st.session_state.messages[-1]["content"]
         response_data = conversation_chat(user_prompt)
         st.session_state.messages.append({"role": "assistant", "content": response_data["response"]})
         st.rerun() # Afficher la réponse de l'assistant


# --- Onglet Suivi Candidatures ---
with tab2:
    st.title("🎯 Tableau de Suivi des Candidatures")

    # --- Fonctions de Gestion des Candidatures ---
    def supprimer_candidature(index):
        if 0 <= index < len(st.session_state.candidatures):
            st.session_state.candidatures.pop(index)
        st.session_state.confirming_delete_index = None
        st.rerun()

    def set_candidature_a_modifier(index):
        st.session_state.candidature_a_modifier = index
        st.session_state.confirming_delete_index = None
        st.rerun()

    def modifier_candidature(index, entreprise, poste, type_candidature, statut, date):
        if 0 <= index < len(st.session_state.candidatures):
            st.session_state.candidatures[index].update({
                "Entreprise": entreprise, "Poste": poste, "Type": type_candidature,
                "Statut": statut, "Date": date.strftime("%Y-%m-%d")
            })
        st.session_state.candidature_a_modifier = None
        st.rerun()

    def afficher_formulaire_modification(index, candidature):
        with st.form(f"modification_candidature_{index}"):
            st.markdown("### ✏️ Modifier la Candidature")
            entreprise = st.text_input("Entreprise / Établissement", value=candidature["Entreprise"])
            poste = st.text_input("Poste / Formation", value=candidature["Poste"])
            type_opts = ["Stage", "Alternance", "Emploi", "Formation"]
            statut_opts = ["En cours", "Accepté", "Refusé"]
            type_index = type_opts.index(candidature["Type"]) if candidature["Type"] in type_opts else 0
            statut_index = statut_opts.index(candidature["Statut"]) if candidature["Statut"] in statut_opts else 0
            type_candidature = st.selectbox("Type", type_opts, index=type_index)
            statut = st.selectbox("Statut", statut_opts, index=statut_index)
            date_val = datetime.today()
            try:
                date_val = datetime.strptime(candidature["Date"], "%Y-%m-%d")
            except ValueError: pass
            date = st.date_input("Date de candidature", date_val)

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 Enregistrer"):
                    if entreprise and poste:
                        modifier_candidature(index, entreprise, poste, type_candidature, statut, date)
                    else: st.warning("Veuillez remplir l'entreprise et le poste.")
            with col2:
                if st.form_submit_button("❌ Annuler"):
                    st.session_state.candidature_a_modifier = None
                    st.rerun()

    def afficher_tableau():
        st.markdown("## 📋 Suivi des Candidatures")
        if not st.session_state.candidatures:
            st.info("Aucune candidature enregistrée.")
            return

        df = pd.DataFrame(st.session_state.candidatures)
        col1, col2 = st.columns(2)
        with col1: statut_filter = st.selectbox("Filtrer par statut", ["Tous"] + df["Statut"].unique().tolist())
        with col2: type_filter = st.selectbox("Filtrer par type", ["Tous"] + df["Type"].unique().tolist())

        filtered_df = df.copy()
        if statut_filter != "Tous": filtered_df = filtered_df[filtered_df["Statut"] == statut_filter]
        if type_filter != "Tous": filtered_df = filtered_df[filtered_df["Type"] == type_filter]

        st.markdown("### Liste des candidatures")
        indices_originaux = filtered_df.index.tolist() # Garder les index originaux du DataFrame complet

        for original_index in indices_originaux: # Itérer sur les index originaux
            row = st.session_state.candidatures[original_index]

            if st.session_state.candidature_a_modifier == original_index:
                afficher_formulaire_modification(original_index, row)

            with st.expander(f"{row['Entreprise']} - {row['Poste']} ({row['Statut']})"):
                col_details, col_edit, col_delete = st.columns([3, 1, 2])
                with col_details:
                    st.write(f"**Type:** {row['Type']}")
                    st.write(f"**Date:** {row['Date']}")
                    st.write(f"**Statut:** {row['Statut']}")
                with col_edit:
                    if st.button("✏️ Modifier", key=f"edit_{original_index}"):
                        set_candidature_a_modifier(original_index)
                with col_delete:
                    if st.session_state.confirming_delete_index == original_index:
                         st.markdown('<div class="delete-confirmation">Confirmer la suppression ?</div>', unsafe_allow_html=True)
                         col_confirm, col_cancel = st.columns(2)
                         with col_confirm:
                              if st.button("Oui, Supprimer", key=f"confirm_delete_{original_index}", type="primary"):
                                   supprimer_candidature(original_index)
                         with col_cancel:
                              if st.button("Annuler", key=f"cancel_delete_{original_index}"):
                                   st.session_state.confirming_delete_index = None
                                   st.rerun()
                    else:
                         if st.button("🗑️ Supprimer", key=f"delete_{original_index}"):
                              st.session_state.confirming_delete_index = original_index
                              st.rerun()

        st.markdown("### Vue d'ensemble (Filtrée)")
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

    def ajouter_candidature():
        with st.form("ajout_candidature"):
            st.markdown("### ➕ Ajouter une Candidature")
            entreprise = st.text_input("Entreprise / Établissement")
            poste = st.text_input("Poste / Formation")
            type_candidature = st.selectbox("Type", ["Stage", "Alternance", "Emploi", "Formation"])
            statut = st.selectbox("Statut", ["En cours", "Accepté", "Refusé"])
            date = st.date_input("Date de candidature", datetime.today())
            submit = st.form_submit_button("Ajouter")

            if submit:
                if entreprise and poste:
                    st.session_state.candidatures.append({
                        "Entreprise": entreprise, "Poste": poste, "Type": type_candidature,
                        "Statut": statut, "Date": date.strftime("%Y-%m-%d")
                    })
                    st.success("Candidature ajoutée !")
                    st.session_state.confirming_delete_index = None
                    st.rerun()
                else:
                    st.warning("Veuillez remplir l'entreprise et le poste.")

    # --- Affichage de l'onglet Suivi ---
    ajouter_candidature()
    afficher_tableau()
