import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vérification de la clé API OpenAI

OPENAI_API_KEY = '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'
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("Clé API OpenAI non trouvée dans les variables d'environnement.")

CHROMA_DB_PATH = "./chroma_db"
PDF_DIRECTORY = "data/pdfs"
MODEL_NAME = "gpt-4o"

def load_and_index_pdfs(directory, persist_directory):
    logger.info(f"Chargement et indexation des PDFs depuis: {directory}")
    if not os.path.exists(directory):
        logger.warning(f"Le dossier {directory} n'existe pas. Aucun PDF ne sera chargé.")
        return None
    try:
        loader = DirectoryLoader(directory, loader_cls=PyPDFLoader, show_progress=True)
        documents = loader.load()
        if not documents:
            logger.warning("Aucun document trouvé ou chargé.")
            return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        logger.info(f"Nombre de chunks créés: {len(texts)}")
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        logger.info(f"Base de données vectorielle créée/mise à jour dans: {persist_directory}")
        return db
    except Exception as e:
        logger.error(f"Erreur lors du chargement/indexation des PDFs: {e}")
        return None

# Initialisation de la base vectorielle
vector_db = None
try:
    if os.path.exists(CHROMA_DB_PATH):
        logger.info(f"Chargement de la base vectorielle existante depuis: {CHROMA_DB_PATH}")
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())
    elif os.path.exists(PDF_DIRECTORY):
        logger.info("Base vectorielle non trouvée, tentative de création à partir des PDFs...")
        vector_db = load_and_index_pdfs(PDF_DIRECTORY, CHROMA_DB_PATH)
        if vector_db is None:
            logger.error("Impossible de créer la base vectorielle.")
    else:
        logger.error(f"Les dossiers {CHROMA_DB_PATH} et {PDF_DIRECTORY} n'existent pas.")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation de ChromaDB: {e}")
    vector_db = None

# Initialisation du LLM
llm = None
try:
    llm = ChatOpenAI(model_name=MODEL_NAME)
    logger.info("Modèle LLM initialisé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation du LLM: {e}. Vérifiez votre clé API et la connexion.")
    llm = None

# Définition des prompts
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    Tu es un assistant spécialisé en orientation académique et professionnelle en France.
    Commence par recueillir les informations suivantes auprès de l'utilisateur :
    - Ses intérêts et compétences.
    - Ses préférences d'études (universitaire, alternance, etc.).
    - Son niveau d'études actuel et ses aspirations professionnelles.
    Une fois ces informations collectées, oriente-le en fonction des données disponibles dans le contexte fourni.
    Si une information dépasse ta connaissance ou le contexte, réponds : "Je ne dispose pas d'informations à ce sujet dans les documents fournis."
    Question: {question}
    """
)

template = """Réponds en utilisant uniquement ce contexte :
{context}

Question: {question}
Réponse:
"""
rag_prompt = ChatPromptTemplate.from_template(template)

# Initialisation du Retriever et de la chaîne RAG
chain = None
if vector_db and llm:
    try:
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            llm=llm,
            prompt=QUERY_PROMPT
        )
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Chaîne RAG initialisée avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du retriever/chaîne RAG: {e}")
        chain = None
else:
    logger.error("La base vectorielle ou le LLM n'a pas pu être initialisé. La chaîne RAG est désactivée.")

# Modèles Pydantic pour l'API
class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# Application FastAPI
app = FastAPI(
    title="Bac2Futur Backend API",
    description="API pour l'assistant d'orientation Bac2Futur",
    version="1.0.0"
)

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: QueryRequest):
    if not chain:
        logger.error("Tentative d'appel à /chat alors que la chaîne RAG n'est pas initialisée.")
        raise HTTPException(status_code=503, detail="Le service de chat n'est pas correctement initialisé. Vérifiez les logs du backend.")
    if not request.query:
        raise HTTPException(status_code=400, detail="La requête ne peut pas être vide.")
    
    logger.info(f"Requête reçue: {request.query}")
    try:
        # Si la version asynchrone est disponible, on peut utiliser: response_content = await chain.ainvoke(request.query)
        response_content = chain.invoke(request.query)
        logger.info(f"Réponse générée: {response_content}")
        return ChatResponse(response=response_content)
    except Exception as e:
        logger.error(f"Erreur lors de l'invocation de la chaîne: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne lors du traitement de la requête: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_enabled": chain is not None, "llm_initialized": llm is not None, "db_initialized": vector_db is not None}

if __name__ == "__main__":
    logger.info("Démarrage du serveur FastAPI...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
    logger.info("Serveur FastAPI démarré.")
