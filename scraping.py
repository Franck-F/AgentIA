import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import time
import json
import os
from datetime import datetime

class ParcoursupScraper:
    def __init__(self):
        self.base_url = "https://dossier.parcoursup.fr/Candidat/public/fiches/formations"
        self.output_dir = "db"
        self.create_output_directory()
        
        # Configuration de Selenium
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Mode sans interface graphique
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=options)
        
    def create_output_directory(self):
        """Crée le répertoire de sortie s'il n'existe pas"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def wait_for_element(self, by, value, timeout=30):
        """Attend qu'un élément soit présent sur la page"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            print(f"Timeout en attendant l'élément {value}")
            return None

    def get_formation_details(self, formation_url):
        """Récupère les détails d'une formation"""
        try:
            self.driver.get(formation_url)
            time.sleep(2)  # Attendre le chargement de la page
            
            # Attendre que le contenu principal soit chargé
            main_content = self.wait_for_element(By.CLASS_NAME, "fiche-formation")
            
            if not main_content:
                return None
            
            # Extraire les informations de la formation
            formation_data = {
                "titre": self.get_text_safe(".titre-formation"),
                "etablissement": self.get_text_safe(".etablissement-formation"),
                "localisation": self.get_text_safe(".localisation-formation"),
                "capacite": self.get_text_safe(".capacite-formation"),
                "taux_acces": self.get_text_safe(".taux-acces"),
                "frais_scolarite": self.get_text_safe(".frais-scolarite"),
                "description": self.get_text_safe(".description-formation"),
                "prerequis": self.get_text_safe(".prerequis-formation"),
                "date_maj": datetime.now().strftime("%Y-%m-%d")
            }
            
            return formation_data
            
        except Exception as e:
            print(f"Erreur lors de la récupération des détails de la formation: {str(e)}")
            return None

    def get_text_safe(self, selector):
        """Récupère le texte d'un élément de manière sécurisée"""
        try:
            element = self.driver.find_element(By.CSS_SELECTOR, selector)
            return element.text.strip()
        except:
            return ""

    def search_formations(self, keyword):
        """Recherche des formations par mot-clé"""
        try:
            search_url = f"{self.base_url}?q={keyword}"
            self.driver.get(search_url)
            
            # Attendre que les résultats soient chargés
            results = self.wait_for_element(By.CLASS_NAME, "resultats-recherche")
            if not results:
                return []
            
            # Récupérer les liens vers les formations
            formation_links = self.driver.find_elements(By.CSS_SELECTOR, ".formation-link")
            formations = []
            
            # Limiter à 10 formations pour l'exemple
            for link in formation_links[:10]:
                formation_url = link.get_attribute("href")
                formation_data = self.get_formation_details(formation_url)
                if formation_data:
                    formations.append(formation_data)
                    
            return formations
            
        except Exception as e:
            print(f"Erreur lors de la recherche des formations: {str(e)}")
            return []

    def save_to_json(self, data, filename):
        """Sauvegarde les données au format JSON"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    def save_to_csv(self, data, filename):
        """Sauvegarde les données au format CSV"""
        df = pd.DataFrame(data)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')

    def scrape_and_save(self, keywords):
        """Scrape les formations pour chaque mot-clé et sauvegarde les résultats"""
        all_formations = []
        
        for keyword in keywords:
            print(f"Recherche des formations pour: {keyword}")
            formations = self.search_formations(keyword)
            all_formations.extend(formations)
            
        if all_formations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_to_json(all_formations, f"formations_{timestamp}.json")
            self.save_to_csv(all_formations, f"formations_{timestamp}.csv")
            print(f"Données sauvegardées dans le dossier {self.output_dir}")
        
    def close(self):
        """Ferme le navigateur"""
        self.driver.quit()

def main():
    # Liste des mots-clés de recherche
    keywords = [
        "informatique",
        "développement web",
        "data science",
        "intelligence artificielle",
        "cybersécurité"
    ]
    
    try:
        scraper = ParcoursupScraper()
        scraper.scrape_and_save(keywords)
    except Exception as e:
        print(f"Erreur lors du scraping: {str(e)}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()
