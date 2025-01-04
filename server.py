#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval chain.

Follow the reference here:

https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain

To run this example, you will need to install the following packages:
pip install langchain openai faiss-cpu tiktoken
"""  # noqa: F401
import os
from dotenv import load_dotenv

from operator import itemgetter
from typing import List, Tuple

from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from langserve import add_routes

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

_TEMPLATE = """Dada a seguinte história de conversação e uma pergunta de acompanhamento, reescreva a pergunta como uma pergunta independente no seu idioma original.

Histórico de Conversa:
{chat_history}
Pergunta de Acompanhamento: {question}
Pergunta Independente:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE =     """Responda à pergunta com base apenas no seguinte contexto:
{context}
Pergunta: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer



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

# Carregar os documentos
documents = load_documents()
vectorstore = create_vector_store(documents)

'''vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)'''

# Configurar o pipeline de RAG
retriever = vectorstore.as_retriever()

_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)