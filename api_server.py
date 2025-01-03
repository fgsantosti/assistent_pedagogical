import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langserve import add_routes
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Templates de prompt
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Dada a seguinte história de conversação e uma pergunta de acompanhamento, reescreva a pergunta como uma pergunta independente no seu idioma original.

Histórico de Conversa:
{chat_history}
Pergunta de Acompanhamento: {question}
Pergunta Independente:"""
)

ANSWER_PROMPT = PromptTemplate.from_template(
    """Responda à pergunta com base apenas no seguinte contexto:
{context}

Pergunta: {question}
"""
)

# Classe para histórico de conversação
class ChatHistory(BaseModel):
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str

# Função para configurar o modelo OpenAI
def setup_openai_model():
    return ChatOpenAI(
        model="gpt-4",
        temperature=0,  # Controle da criatividade
        openai_api_key=OPENAI_API_KEY,
    )

# Função para carregar documentos de exemplo
def load_documents():
    """Carrega documentos de exemplo para criar um vetor FAISS."""
    print("Carregando documentos...")
    return ["Exemplo de texto para demonstração do RAG."]

# Função para criar vetor FAISS
def create_vector_store(documents):
    print("Criando índice vetorial...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store

# Função para configurar a pipeline de RAG
def create_rag_pipeline(vector_store):
    print("Configurando pipeline RAG...")
    retriever = vector_store.as_retriever()
    model = setup_openai_model()
    pipeline = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        qa_prompt=ANSWER_PROMPT,
    )
    return pipeline

# Configuração do servidor FastAPI
app = FastAPI(
    title="LangChain RAG Server",
    version="1.0",
    description="Servidor RAG baseado em LangChain com FastAPI.",
)

# Inicialização do vetor FAISS e pipeline
documents = load_documents()
vector_store = create_vector_store(documents)
rag_pipeline = create_rag_pipeline(vector_store)

# Configura rotas usando LangServe
add_routes(app, rag_pipeline.with_types(input_type=ChatHistory))

# Iniciar o servidor
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
