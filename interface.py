import streamlit as st
import pandas as pd
from datetime import datetime

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
st.set_page_config(page_title="Suivi des Candidatures", layout="wide")
st.title("🎯 Tableau de Suivi des Candidatures")

ajouter_candidature()
afficher_tableau()
