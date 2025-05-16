import streamlit as st
import pandas as pd
from datetime import datetime
import os
from back import (
    chain, 
    needs_clarification, 
    generate_clarifying_questions,
    generate_complete_response,
    detect_greeting,
    clean_response
)
from document_generator import DocumentGenerator
from model_trainer import ModelTrainer

# Configuration de la page
st.set_page_config(
    page_title="Assistant IA et Suivi des Candidatures",
    page_icon="🤖",
    layout="wide"
)

# Initialisation des états de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "candidatures" not in st.session_state:
    st.session_state.candidatures = []
if "candidature_a_modifier" not in st.session_state:
    st.session_state.candidature_a_modifier = None
if "document_generator" not in st.session_state:
    st.session_state.document_generator = DocumentGenerator()
if "model_trainer" not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()

# Sidebar pour les informations et paramètres
with st.sidebar:
    st.header("📚 Informations")
    st.markdown("""
    Cet assistant vous aide dans votre orientation scolaire et professionnelle.
    
    ### 📝 Il peut vous aider pour :
    - Le choix de vos études
    - La rédaction de lettres de motivation
    - La préparation de CV
    - Les candidatures Parcoursup
    - Le suivi de vos candidatures
    """)
    
    # Création et affichage du répertoire des PDFs
    pdf_directory = os.path.join("data", "pdfs")
    os.makedirs(pdf_directory, exist_ok=True)
    
    st.header("📑 Documents")
    uploaded_file = st.file_uploader("Ajouter un document PDF", type=['pdf'])
    if uploaded_file:
        with open(os.path.join(pdf_directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"Document {uploaded_file.name} ajouté avec succès !")
    
    pdfs = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    if pdfs:
        st.write("Documents chargés :")
        for pdf in pdfs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {pdf}")
            with col2:
                if st.button("🗑️", key=f"del_{pdf}"):
                    os.remove(os.path.join(pdf_directory, pdf))
                    st.rerun()
    else:
        st.info("Aucun document PDF chargé. Utilisez le bouton ci-dessus pour ajouter des documents.")

# Fonction pour générer les documents d'une candidature
def generer_documents_candidature(candidature):
    # Préparer les données pour le CV
    cv_data = {
        "nom": st.session_state.get("nom", ""),
        "prenom": st.session_state.get("prenom", ""),
        "email": st.session_state.get("email", ""),
        "telephone": st.session_state.get("telephone", ""),
        "adresse": st.session_state.get("adresse", ""),
        "formations": st.session_state.get("formations", []),
        "experiences": st.session_state.get("experiences", []),
        "competences": st.session_state.get("competences", [])
    }
    
    # Préparer les données pour la lettre
    lettre_data = {
        "expediteur": {
            "nom": st.session_state.get("nom", ""),
            "prenom": st.session_state.get("prenom", ""),
            "email": st.session_state.get("email", ""),
            "telephone": st.session_state.get("telephone", ""),
            "adresse": st.session_state.get("adresse", "")
        },
        "destinataire": {
            "entreprise": candidature["Entreprise"],
            "contact": "",
            "adresse": ""
        },
        "lieu": st.session_state.get("ville", ""),
        "date": datetime.now().strftime("%d/%m/%Y"),
        "objet": f"Candidature pour le poste de {candidature['Poste']}",
        "contenu": st.session_state.get("lettre_contenu", "")
    }
    
    # Créer le dossier de sortie
    output_dir = os.path.join("output", "candidatures", 
                             f"{candidature['Entreprise']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Générer les documents
    result = st.session_state.document_generator.generate_documents_for_candidature(
        cv_data, lettre_data, output_dir
    )
    
    return result

# Fonction pour afficher le formulaire de profil
def afficher_profil():
    st.markdown("## 👤 Mon Profil")
    
    with st.expander("Informations Personnelles"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.nom = st.text_input("Nom", st.session_state.get("nom", ""))
            st.session_state.prenom = st.text_input("Prénom", st.session_state.get("prenom", ""))
            st.session_state.email = st.text_input("Email", st.session_state.get("email", ""))
        with col2:
            st.session_state.telephone = st.text_input("Téléphone", st.session_state.get("telephone", ""))
            st.session_state.adresse = st.text_area("Adresse", st.session_state.get("adresse", ""))
            st.session_state.ville = st.text_input("Ville", st.session_state.get("ville", ""))
    
    with st.expander("Formation"):
        st.session_state.formations = st.session_state.get("formations", [])
        for i, formation in enumerate(st.session_state.formations):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{formation['diplome']}** - {formation['etablissement']}")
                with col2:
                    st.write(formation['periode'])
                with col3:
                    if st.button("🗑️", key=f"del_formation_{i}"):
                        st.session_state.formations.pop(i)
                        st.rerun()
        
        if st.button("➕ Ajouter une formation"):
            st.session_state.formations.append({
                "diplome": "",
                "etablissement": "",
                "periode": ""
            })
            st.rerun()
    
    with st.expander("Expérience Professionnelle"):
        st.session_state.experiences = st.session_state.get("experiences", [])
        for i, experience in enumerate(st.session_state.experiences):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{experience['poste']}** - {experience['entreprise']}")
                with col2:
                    st.write(experience['periode'])
                with col3:
                    if st.button("🗑️", key=f"del_experience_{i}"):
                        st.session_state.experiences.pop(i)
                        st.rerun()
        
        if st.button("➕ Ajouter une expérience"):
            st.session_state.experiences.append({
                "poste": "",
                "entreprise": "",
                "periode": "",
                "taches": []
            })
            st.rerun()
    
    with st.expander("Compétences"):
        st.session_state.competences = st.text_area(
            "Liste des compétences (une par ligne)",
            value="\n".join(st.session_state.get("competences", [])),
            height=150
        ).split("\n")

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
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"**Type:** {row['Type']}")
                st.write(f"**Date:** {row['Date']}")
            with col2:
                if st.button("✏️ Modifier", key=f"mod_{index}"):
                    set_candidature_a_modifier(index)
            with col3:
                if st.button("🗑️ Supprimer", key=f"del_{index}"):
                    supprimer_candidature(index)
            with col4:
                if st.button("📄 Générer Documents", key=f"doc_{index}"):
                    with st.spinner("Génération des documents en cours..."):
                        result = generer_documents_candidature(row)
                        if result["success"]:
                            st.success("Documents générés avec succès !")
                            st.markdown(f"📁 [CV]({result['cv_path']}) | 📝 [Lettre de motivation]({result['lettre_path']})")
                        else:
                            st.error("Erreur lors de la génération des documents.")

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

# Affichage de l'interface principale
tabs = st.tabs(["💬 Assistant IA", "👤 Mon Profil", "📋 Candidatures", "⚙️ Paramètres"])

with tabs[0]:  # Assistant IA
    # Affichage du titre de l'Assistant IA
    st.title("💬 Assistant IA d'Orientation")

    # Affichage de l'historique des messages de l'assistant IA
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de saisie utilisateur pour l'IA
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajout du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Obtention de la réponse de l'assistant
        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                # Vérifier si c'est une salutation
                greeting = detect_greeting(prompt)
                if greeting:
                    response = f"{greeting} ! Je suis votre assistant virtuel spécialisé dans l'orientation. Comment puis-je vous aider aujourd'hui ?"
                
                # Vérifier si la question nécessite des clarifications
                elif needs_clarification(prompt):
                    clarifying_questions = generate_clarifying_questions(prompt)
                    response = "Pour mieux vous aider, j'aurais besoin de quelques précisions :\n\n"
                    for question in clarifying_questions:
                        response += f"- {question}\n"
                
                # Générer une réponse complète
                else:
                    try:
                        # Utiliser la chaîne RAG pour les réponses basées sur les documents
                        rag_response = chain.invoke(prompt)
                        # Générer une réponse complète en tenant compte du contexte de la conversation
                        complete_response = generate_complete_response(
                            prompt, 
                            st.session_state.conversation_history
                        )
                        # Combiner et nettoyer la réponse
                        response = clean_response(f"{rag_response}\n\n{complete_response}")
                    except Exception as e:
                        response = "Je suis désolé, mais j'ai rencontré une erreur lors du traitement de votre demande. Pourriez-vous reformuler votre question ?"
                        st.error(f"Erreur : {str(e)}")

                # Afficher la réponse
                st.markdown(response)

                # Mettre à jour l'historique de la conversation
                st.session_state.conversation_history.append(prompt)
                st.session_state.conversation_history.append(response)
                
                # Ajouter la réponse aux messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

with tabs[1]:  # Mon Profil
    afficher_profil()

with tabs[2]:  # Candidatures
    ajouter_candidature()
    afficher_tableau()

with tabs[3]:  # Paramètres
    st.markdown("## ⚙️ Paramètres")
    
    with st.expander("🔄 Fine-tuning du modèle"):
        if st.button("Lancer le fine-tuning"):
            with st.spinner("Préparation et lancement du fine-tuning..."):
                # Préparer les données de conversation pour le fine-tuning
                conversations = []
                for i in range(0, len(st.session_state.conversation_history), 2):
                    if i + 1 < len(st.session_state.conversation_history):
                        conversations.append({
                            "messages": [
                                {"role": "user", "content": st.session_state.conversation_history[i]},
                                {"role": "assistant", "content": st.session_state.conversation_history[i + 1]}
                            ]
                        })
                
                # Lancer le fine-tuning
                result = st.session_state.model_trainer.fine_tune_model(
                    conversations,
                    "output/training"
                )
                
                if result["status"] == "succeeded":
                    st.success(f"Fine-tuning réussi ! Nouveau modèle : {result['fine_tuned_model']}")
                else:
                    st.error(f"Erreur lors du fine-tuning : {result.get('error', 'Erreur inconnue')}")
    
    with st.expander("📁 Gestion des documents"):
        st.markdown("### Templates de documents")
        uploaded_cv = st.file_uploader("Template de CV personnalisé", type=["html"])
        if uploaded_cv:
            with open(os.path.join("templates", "cv_template.html"), "wb") as f:
                f.write(uploaded_cv.getvalue())
            st.success("Template de CV mis à jour !")
        
        uploaded_lettre = st.file_uploader("Template de lettre personnalisé", type=["html"])
        if uploaded_lettre:
            with open(os.path.join("templates", "lettre_template.html"), "wb") as f:
                f.write(uploaded_lettre.getvalue())
            st.success("Template de lettre mis à jour !")
