
# Chatbot IA Santé Mentale – Projet RAG avec LangChain

## 🧠 Description

Ce projet consiste à créer un chatbot conversationnel spécialisé dans le domaine de la santé mentale.
Il est basé sur une architecture RAG (Retrieval-Augmented Generation) combinant un LLM (LLaMA 3 via Groq)
avec une base de documents vectorisée pour fournir des réponses précises, fiables et contextuelles.

## 🧰 Technologies utilisées

- Python 3.13
- LangChain
- Gradio
- ChromaDB
- HuggingFace Transformers
- LLaMA 3 (via ChatGroq)

## 🔧 Installation

```bash
git clone https://github.com/bahakechich/chatbot-sante-mentale.git
cd chatbot-sante-mentale
python -m venv venv
.env\Scriptsctivate
pip install -r requirements.txt
```

## 🔐 Configuration

Créez un fichier `.env` à la racine du projet avec :
```
GROQ_API_KEY=ta_cle_groq_ici
```

## 🚀 Lancement du projet

```bash
python chatbot_ui.py
```

Puis ouvrez : [http://127.0.0.1:7860](http://127.0.0.1:7860)

## 💬 Fonctionnalités

- Réponses basées sur un vrai contenu santé mentale (`sante_mentale.txt`)
- Recherche sémantique avec embeddings + Chroma
- Mémoire conversationnelle intégrée
- Interface conviviale avec Gradio
- Pas d’hallucination, contenu contrôlé

## 🧪 Exemples de questions

- Quels sont les signes de dépression ?
- Qui consulter pour une anxiété sévère ?
- Comment prendre soin de sa santé mentale ?

