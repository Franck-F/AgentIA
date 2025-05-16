import os
from typing import Dict, Any
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime
from fpdf import FPDF
import jinja2
import pdfkit

class DocumentGenerator:
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = templates_dir
        self._ensure_templates_dir()
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir)
        )

    def _ensure_templates_dir(self):
        """Crée le répertoire des templates s'il n'existe pas"""
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Création des templates par défaut s'ils n'existent pas
        self._create_default_cv_template()
        self._create_default_lettre_template()

    def _create_default_cv_template(self):
        """Crée un template de CV par défaut"""
        template_path = os.path.join(self.templates_dir, "cv_template.html")
        if not os.path.exists(template_path):
            template = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .section { margin: 20px 0; }
                    .section-title { 
                        color: #2c3e50;
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 5px;
                    }
                    .experience-item { margin: 15px 0; }
                    .experience-title { font-weight: bold; }
                    .experience-date { color: #7f8c8d; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{{ nom }} {{ prenom }}</h1>
                    <p>{{ email }} | {{ telephone }}</p>
                    <p>{{ adresse }}</p>
                </div>

                <div class="section">
                    <h2 class="section-title">Formation</h2>
                    {% for formation in formations %}
                    <div class="experience-item">
                        <div class="experience-title">{{ formation.diplome }}</div>
                        <div class="experience-date">{{ formation.periode }}</div>
                        <div>{{ formation.etablissement }}</div>
                    </div>
                    {% endfor %}
                </div>

                <div class="section">
                    <h2 class="section-title">Expérience Professionnelle</h2>
                    {% for experience in experiences %}
                    <div class="experience-item">
                        <div class="experience-title">{{ experience.poste }}</div>
                        <div class="experience-date">{{ experience.periode }}</div>
                        <div>{{ experience.entreprise }}</div>
                        <ul>
                        {% for tache in experience.taches %}
                            <li>{{ tache }}</li>
                        {% endfor %}
                        </ul>
                    </div>
                    {% endfor %}
                </div>

                <div class="section">
                    <h2 class="section-title">Compétences</h2>
                    <ul>
                    {% for competence in competences %}
                        <li>{{ competence }}</li>
                    {% endfor %}
                    </ul>
                </div>
            </body>
            </html>
            """
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template)

    def _create_default_lettre_template(self):
        """Crée un template de lettre de motivation par défaut"""
        template_path = os.path.join(self.templates_dir, "lettre_template.html")
        if not os.path.exists(template_path):
            template = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .expediteur { margin-bottom: 30px; }
                    .destinataire { margin-bottom: 30px; }
                    .date { margin-bottom: 30px; text-align: right; }
                    .objet { margin-bottom: 30px; }
                    .contenu { margin-bottom: 30px; text-align: justify; }
                    .signature { margin-top: 50px; }
                </style>
            </head>
            <body>
                <div class="expediteur">
                    {{ expediteur.nom }} {{ expediteur.prenom }}<br>
                    {{ expediteur.adresse }}<br>
                    {{ expediteur.email }}<br>
                    {{ expediteur.telephone }}
                </div>

                <div class="destinataire">
                    {{ destinataire.entreprise }}<br>
                    {% if destinataire.contact %}À l'attention de {{ destinataire.contact }}<br>{% endif %}
                    {{ destinataire.adresse }}
                </div>

                <div class="date">
                    {{ lieu }}, le {{ date }}
                </div>

                <div class="objet">
                    <strong>Objet :</strong> {{ objet }}
                </div>

                <div class="contenu">
                    {{ contenu | safe }}
                </div>

                <div class="signature">
                    {{ expediteur.nom }} {{ expediteur.prenom }}
                </div>
            </body>
            </html>
            """
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template)

    def generate_cv(self, data: Dict[str, Any], output_path: str) -> bool:
        """Génère un CV au format PDF"""
        try:
            template = self.jinja_env.get_template("cv_template.html")
            html_content = template.render(**data)
            
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
            }
            
            pdfkit.from_string(html_content, output_path, options=options)
            return True
        except Exception as e:
            print(f"Erreur lors de la génération du CV : {str(e)}")
            return False

    def generate_lettre_motivation(self, data: Dict[str, Any], output_path: str) -> bool:
        """Génère une lettre de motivation au format PDF"""
        try:
            template = self.jinja_env.get_template("lettre_template.html")
            html_content = template.render(**data)
            
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
            }
            
            pdfkit.from_string(html_content, output_path, options=options)
            return True
        except Exception as e:
            print(f"Erreur lors de la génération de la lettre : {str(e)}")
            return False

    def generate_documents_for_candidature(self, 
                                        cv_data: Dict[str, Any],
                                        lettre_data: Dict[str, Any],
                                        output_dir: str) -> Dict[str, str]:
        """Génère le CV et la lettre de motivation pour une candidature"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv_path = os.path.join(output_dir, f"cv_{timestamp}.pdf")
        lettre_path = os.path.join(output_dir, f"lettre_{timestamp}.pdf")
        
        cv_success = self.generate_cv(cv_data, cv_path)
        lettre_success = self.generate_lettre_motivation(lettre_data, lettre_path)
        
        return {
            "cv_path": cv_path if cv_success else None,
            "lettre_path": lettre_path if lettre_success else None,
            "success": cv_success and lettre_success
        }
