import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langserve import launch_langserve
from docx import Document

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Função para carregar documentos
def load_documents():
    print("Carregando documentos...")
    documents = []
    folder_path = "./documents"

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

# Configurar o índice vetorial
def create_vector_store(documents):
    print("Criando índice vetorial...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# Configurar a cadeia RAG
def create_rag_pipeline(vector_store):
    print("Configurando pipeline RAG...")
    retriever = vector_store.as_retriever()
    model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
    pipeline = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
    )
    return pipeline

# Inicializar a API com LangServe
if __name__ == "__main__":
    # Carregar documentos e criar a cadeia RAG
    documents = load_documents()
    vector_store = create_vector_store(documents)
    rag_pipeline = create_rag_pipeline(vector_store)

    # Inicializar o LangServe com a cadeia configurada
    print("Inicializando API REST...")
    launch_langserve(rag_pipeline, port=8000)