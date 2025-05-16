import os
from typing import List, Dict, Any
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredURLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import trafilatura
import requests
from bs4 import BeautifulSoup
import docx2txt
from tqdm import tqdm

class DocumentProcessor:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.supported_extensions = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1000,
            embedding_ctx_length=8191
        )

    def create_directory_structure(self):
        """Crée la structure de répertoires nécessaire"""
        directories = [
            os.path.join(self.base_dir, d) 
            for d in ["pdfs", "docs", "txt", "web_content"]
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        return directories

    def process_web_content(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Traite et sauvegarde le contenu des pages web"""
        documents = []
        for url in tqdm(urls, desc="Traitement des pages web"):
            try:
                # Extraction du contenu avec trafilatura
                downloaded_content = trafilatura.fetch_url(url)
                content = trafilatura.extract(downloaded_content)
                
                if content:
                    # Sauvegarde dans un fichier texte
                    filename = url.split("/")[-1].replace(".", "_") + ".txt"
                    filepath = os.path.join(self.base_dir, "web_content", filename)
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    documents.append({
                        "content": content,
                        "metadata": {"source": url, "type": "web"}
                    })
            except Exception as e:
                print(f"Erreur lors du traitement de {url}: {str(e)}")
        
        return documents

    def load_documents(self) -> List[Dict[str, Any]]:
        """Charge tous les documents supportés"""
        documents = []
        
        # Création des répertoires
        self.create_directory_structure()
        
        # Chargement des documents par type
        for root, _, files in os.walk(self.base_dir):
            for file in tqdm(files, desc=f"Traitement des fichiers dans {root}"):
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                try:
                    if ext in self.supported_extensions:
                        loader_class = self.supported_extensions[ext]
                        loader = loader_class(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                except Exception as e:
                    print(f"Erreur lors du chargement de {file_path}: {str(e)}")
        
        return documents

    def create_vector_store(self, documents: List[Dict[str, Any]]) -> Chroma:
        """Crée une base de données vectorielle à partir des documents"""
        texts = self.text_splitter.split_documents(documents)
        return Chroma.from_documents(
            texts,
            self.embeddings,
            persist_directory="./chroma_db"
        )

    def process_all(self, urls: List[str] = None) -> Chroma:
        """Traite tous les documents et crée la base de données vectorielle"""
        documents = self.load_documents()
        
        if urls:
            web_documents = self.process_web_content(urls)
            documents.extend(web_documents)
        
        return self.create_vector_store(documents)

    def add_document(self, file_path: str) -> bool:
        """Ajoute un nouveau document à la base existante"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.supported_extensions:
                return False
            
            loader_class = self.supported_extensions[ext]
            loader = loader_class(file_path)
            documents = loader.load()
            
            # Mise à jour de la base vectorielle
            vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings
            )
            texts = self.text_splitter.split_documents(documents)
            vector_store.add_documents(texts)
            return True
        except Exception as e:
            print(f"Erreur lors de l'ajout du document {file_path}: {str(e)}")
            return False
