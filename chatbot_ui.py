# chatbot_ui.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import gradio as gr

# Charger la clé GROQ
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# LLM (LLaMA 3 via Groq)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192",
    temperature=0.5
)

# Chargement du document santé mentale
loader = TextLoader("sante_mentale.txt", encoding="utf-8")
documents = loader.load()

# Découper le texte en petits morceaux
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Créer les embeddings et la base Chroma
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="./chroma_db")
retriever = vectordb.as_retriever()

# ➕ Ajouter la mémoire de conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Créer la chaîne RAG avec mémoire
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    memory=memory
)

# Fonction chatbot Gradio
def chatbot(query, history):
    result = rag_chain.run(query)
    return result

# Interface utilisateur Gradio avec avatar, intro, mémoire
with gr.Blocks() as iface:
    # Titre et avatar
    gr.Markdown("""
    # 🧠 Chatbot Santé Mentale
    <div style="text-align: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" alt="Avatar" width="80"/>
    </div>
    **👋 Bonjour ! Je suis là pour vous écouter et vous conseiller sur les questions de santé mentale.**
    """)

    # Interface de chat
    gr.ChatInterface(
        fn=chatbot,
        chatbot=gr.Chatbot(label="Assistant IA Santé Mentale", type="messages"),
        textbox=gr.Textbox(placeholder="Posez votre question ici...", label="Votre question"),
        submit_btn="Envoyer"
    )

# Lancement
if __name__ == "__main__":
    iface.launch()

#activer environnement .\venv\Scripts\Activate.ps1
#run script pyhton chatbot_ui.py