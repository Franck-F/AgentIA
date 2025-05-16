import os
import json
from typing import List, Dict, Any
import tiktoken
from openai import OpenAI
import time
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, base_model: str = "gpt-3.5-turbo"):
        self.client = OpenAI()
        self.base_model = base_model
        self.encoding = tiktoken.encoding_for_model(base_model)

    def prepare_training_data(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prépare les données pour le fine-tuning"""
        training_data = []
        
        for conv in conversations:
            messages = []
            
            # Ajouter le message système si présent
            if "system" in conv:
                messages.append({
                    "role": "system",
                    "content": conv["system"]
                })
            
            # Ajouter les messages de la conversation
            for msg in conv["messages"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            training_data.append({"messages": messages})
        
        return training_data

    def validate_training_data(self, training_data: List[Dict[str, Any]]) -> bool:
        """Valide le format des données d'entraînement"""
        try:
            for conversation in training_data:
                if "messages" not in conversation:
                    return False
                
                messages = conversation["messages"]
                if not messages:
                    return False
                
                for message in messages:
                    if "role" not in message or "content" not in message:
                        return False
                    
                    if message["role"] not in ["system", "user", "assistant"]:
                        return False
                    
                    if not isinstance(message["content"], str):
                        return False
            
            return True
        except Exception:
            return False

    def count_tokens(self, training_data: List[Dict[str, Any]]) -> int:
        """Compte le nombre total de tokens dans les données d'entraînement"""
        total_tokens = 0
        
        for conversation in training_data:
            for message in conversation["messages"]:
                tokens = self.encoding.encode(message["content"])
                total_tokens += len(tokens)
        
        return total_tokens

    def save_training_data(self, training_data: List[Dict[str, Any]], output_file: str):
        """Sauvegarde les données d'entraînement au format JSONL"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def create_fine_tuning_job(self, training_file_path: str) -> str:
        """Crée et lance un job de fine-tuning"""
        try:
            # Upload du fichier
            with open(training_file_path, 'rb') as f:
                training_file = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )

            # Création du job de fine-tuning
            job = self.client.fine_tuning.jobs.create(
                training_file=training_file.id,
                model=self.base_model
            )

            return job.id
        except Exception as e:
            print(f"Erreur lors de la création du job de fine-tuning : {str(e)}")
            return None

    def monitor_fine_tuning_job(self, job_id: str) -> Dict[str, Any]:
        """Surveille l'état d'un job de fine-tuning"""
        try:
            while True:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                
                status = job.status
                print(f"État du job : {status}")
                
                if status in ["succeeded", "failed"]:
                    return {
                        "status": status,
                        "fine_tuned_model": job.fine_tuned_model if status == "succeeded" else None,
                        "error": job.error if status == "failed" else None
                    }
                
                time.sleep(60)  # Attendre 1 minute avant la prochaine vérification
        except Exception as e:
            print(f"Erreur lors du monitoring du job : {str(e)}")
            return {"status": "error", "error": str(e)}

    def fine_tune_model(self, conversations: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
        """Process complet de fine-tuning"""
        try:
            # Créer le répertoire de sortie
            os.makedirs(output_dir, exist_ok=True)
            
            # Préparer les données
            training_data = self.prepare_training_data(conversations)
            
            # Valider les données
            if not self.validate_training_data(training_data):
                return {"status": "error", "error": "Format des données invalide"}
            
            # Vérifier le nombre de tokens
            token_count = self.count_tokens(training_data)
            if token_count < 100:  # Minimum requis par OpenAI
                return {"status": "error", "error": "Pas assez de données d'entraînement"}
            
            # Sauvegarder les données
            training_file = os.path.join(output_dir, "training_data.jsonl")
            self.save_training_data(training_data, training_file)
            
            # Lancer le fine-tuning
            job_id = self.create_fine_tuning_job(training_file)
            if not job_id:
                return {"status": "error", "error": "Échec de la création du job"}
            
            # Monitorer le job
            result = self.monitor_fine_tuning_job(job_id)
            
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
