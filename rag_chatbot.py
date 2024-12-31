import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from docx import Document

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurando o modelo OpenAI
def setup_openai_model():
    return ChatOpenAI(
        model="gpt-4",
        temperature=0,  # Controle da criatividade das respostas
        openai_api_key=OPENAI_API_KEY,
    )

# Carregar os documentos
def load_documents():
    print("Carregando documentos...")
    documents = []

    # Caminho da pasta onde os documentos estão
    folder_path = "./documents"

    # Iterar por todos os arquivos na pasta
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Carregar arquivos TXT
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        # Carregar arquivos PDF
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        # Carregar arquivos DOCX
        elif file_name.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            documents.append({"text": text, "metadata": {"source": file_name}})

    print(f"Carregado {len(documents)} documentos.")
    return documents

# Criar o índice de vetores para RAG
def create_vector_store(documents):
    print("Criando índice vetorial...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Configurar o pipeline de RAG
def create_rag_pipeline(vector_store):
    print("Configurando pipeline RAG...")
    retriever = vector_store.as_retriever()
    model = setup_openai_model()
    pipeline = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
    )
    return pipeline

# Interface principal do chatbot
def chatbot_interface(pipeline):
    print("Chatbot iniciado. Digite 'sair' para encerrar.")
    chat_history = []
    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() == "sair":
            print("Encerrando o chatbot.")
            break
        response = pipeline.run({"question": user_input, "chat_history": chat_history})
        print(f"Chatbot: {response}")
        chat_history.append((user_input, response))

# Função principal
if __name__ == "__main__":
    documents = load_documents()
    vector_store = create_vector_store(documents)
    rag_pipeline = create_rag_pipeline(vector_store)
    chatbot_interface(rag_pipeline)
