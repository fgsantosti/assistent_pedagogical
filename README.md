## LangChain
Abaixo é apresentado um guia passo a passo para criar um chatbot com **RAG** (Retrieval-Augmented Generation) usando **LangChain**, **OpenAI**, e **Groq**. 

--- 

### **O que é RAG (Retrieval-Augmented Generation)?**
RAG é uma técnica onde um modelo de linguagem (como o GPT) é combinado com uma base de conhecimento externa. Em vez de confiar apenas no modelo para gerar respostas, ele recupera informações de uma base de dados ou documentos, garantindo maior precisão e controle das respostas.

Como documento para o modelo extrair o conhecimento externo irei utilizar a organização didática do Instituto Federal do Piauí - IFPI. Este tem como objetivo da organização didática do Instituto Federal de Educação, Ciência e Tecnologia do Piauí (IFPI) é promover uma educação de excelência direcionada às demandas sociais. Isso inclui oferecer educação profissional e tecnológica em todos os níveis e modalidades, formar e qualificar cidadãos para atuação profissional em diversos setores da economia, desenvolver a educação profissional e tecnológica como um processo educativo e investigativo, promover a integração e a verticalização da educação básica à educação profissional e superior, e orientar sua oferta formativa em benefício da consolidação e fortalecimento dos arranjos produtivos, sociais e culturais locais. 

---

### **Ferramentas que Usaremos**
1. **LangChain**: Um framework que facilita a construção de pipelines para chatbots baseados em grandes modelos de linguagem.
2. **OpenAI**: Fornece o modelo GPT para geração de respostas.
3. **Groq**: Um mecanismo de aceleração para processamento rápido e eficiente.
4. **Base de Dados**: Uma base de conhecimento local, que pode ser um conjunto de arquivos ou uma base de dados.

---

### **Pré-requisitos**
1. Instalar Python (>=3.8).
2. Criar uma conta no OpenAI para obter uma API Key.
3. Ter uma base de dados ou documentos para servir como fonte de informações.
4. Configurar o ambiente para usar o hardware Groq (se disponível).

---

### **Passo 1: Configurando o Ambiente**

---

### 1. **Usar um Ambiente Virtual**
Crie um ambiente virtual e instale os pacotes dentro dele para evitar interferências no sistema.

```bash
python3 -m venv venv
source venv/bin/activate
pip install nome_do_pacote
```

Para sair do ambiente virtual, use:

```bash
deactivate
```

---


Instale as bibliotecas necessárias:
```bash
pip install langchain openai groq python-dotenv
```

Para carregar arquivos em diferentes formatos, como `PDF, DOCX, além de TXT`, podemos usar bibliotecas específicas para cada formato. O LangChain possui suporte nativo para carregar arquivos PDF, mas para carregar arquivos .docx (Word), precisamos usar uma biblioteca como python-docx.

```bash
pip install langchain pypdf python-docx
```


---

### **Passo 2: Configurando Variáveis de Ambiente**
Crie um arquivo `.env` no diretório do projeto para armazenar a chave da API do OpenAI:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

Carregaremos essas variáveis no código usando a biblioteca `dotenv`.

---

### **Passo 3: Estruturando o Código**
Segue um código comentado para o chatbot com RAG.

#### **Arquivo: `rag_chatbot.py`**
```python
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

```

---

### **Passo 4: Preparar Documentos**
Crie uma pasta chamada `documents` e adicione arquivos de texto que serão usados como base de conhecimento. Por exemplo:
- `document1.txt`: Informações gerais sobre a sua área.
- `document2.txt`: Respostas específicas para perguntas frequentes.

---

### **Passo 5: Executando o Chatbot**
Execute o chatbot com o comando:
```bash
python rag_chatbot.py
```

---

### **Possíveis Expansões**
1. **Adicionar Suporte a Novos Formatos**:
   - Adicionar suporte a documentos PDF ou bases de dados SQL.
2. **Deploy**:
   - Implementar uma interface web com **Streamlit** ou **Flask**.
3. **Otimizações com Groq**:
   - Explorar como o Groq pode acelerar consultas em bases de dados grandes.

---

## LangServe

Vamos integrar o aplicativo ao **LangServe**, que facilita a implantação de **runnables** e cadeias do LangChain como uma **API REST**. Isso permitirá que o chatbot seja acessado por meio de chamadas HTTP, expandindo seu uso para aplicações web ou móveis.


[Langserve Documentação](https://python.langchain.com/docs/langserve/)
[Conversational Retriever](https://github.com/langchain-ai/langserve/blob/main/examples/conversational_retrieval_chain/server.py)


---

### **Passo 1: Instalar o LangServe**
O LangServe é parte do ecossistema do LangChain. Instale-o com o comando:
```bash
pip install langserve

pip install fastapi uvicorn

```

---

### **Passo 2: Adaptar o Código para Usar LangServe**
A integração com o LangServe exige criar um arquivo principal onde definimos a cadeia (`chain`) que será disponibilizada como uma API REST.

Aqui está o código atualizado para integração com o LangServe:

#### **Arquivo: `rag_api.py`**
```python
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

```

O código acima implementa um servidor FastAPI para um chatbot baseado em Recuperação e Geração de Respostas (RAG) utilizando a biblioteca LangChain. Aqui está o entendimento geral:

### **Visão Geral**
1. **Carregamento de Variáveis de Ambiente**:
   - Utiliza o `dotenv` para carregar a chave da API da OpenAI (`OPENAI_API_KEY`) do arquivo `.env`.

2. **Carregamento de Documentos**:
   - Lê arquivos `.txt`, `.pdf` e `.docx` da pasta `./documents`, processando-os para serem utilizados na criação do índice vetorial.

3. **Criação de Índice Vetorial**:
   - Utiliza o FAISS para criar um índice vetorial dos documentos processados, permitindo a busca eficiente de informações baseadas em embeddings gerados pelo OpenAI.

4. **Pipeline de Recuperação e Resposta**:
   - Implementa prompts personalizados para:
     - Reformular perguntas de acompanhamento para perguntas independentes.
     - Responder perguntas com base apenas no contexto recuperado dos documentos.
   - Usa `RunnableMap` e outros utilitários do LangChain para compor essa pipeline.

5. **Interface do Usuário (API)**:
   - Cria um aplicativo FastAPI que expõe endpoints como `/invoke`, `/batch`, e `/stream` para interação com o chatbot.

6. **Execução do Servidor**:
   - Utiliza `uvicorn` para executar o servidor localmente na porta `8000`.


---

### **O que este código faz?**
1. **Carregamento de Documentos**:
   - Lê arquivos nos formatos `.txt`, `.pdf`, e `.docx`.
2. **Criação do Índice Vetorial**:
   - Utiliza `FAISS` para armazenar embeddings dos documentos.
3. **Configuração do RAG**:
   - Configura a cadeia RAG utilizando o modelo GPT-4 da OpenAI.
4. **Lançamento da API REST**:
   - Usa `LangServe` para expor a cadeia como uma API REST na porta 8000.

---

### **Passo 3: Executar a API**
Para iniciar a API, execute o arquivo:
```bash
python rag_api.py
```

---

### **Testando a API**
Com a API em execução, você pode testar suas funcionalidades usando ferramentas como **Postman**, **cURL** ou até mesmo navegadores. Segue um exemplo de teste com `curl`:

#### **Exemplo de Requisição com Python usando `requests`**
```bash
import requests

inputs = {"input": {"question": "Qual o objetivo da organização didática?", "chat_history": []}}
response = requests.post("http://localhost:8000/invoke", json=inputs)

response.json()
```

output
```bash
{'output': 'O objetivo da organização didática é reger as atividades e decisões didático-pedagógicas desenvolvidas no Instituto Federal de Educação, Ciência e Tecnologia do Piauí, observando as disposições legais que regulamentam a educação no Brasil.',
 'metadata': {'run_id': '8c0d339c-0481-4aa5-aa6b-514f37d02131',
  'feedback_tokens': []}}
```


```bash
import requests

inputs = {"input": {"question": "O que deve constar em um plano de disciplina?", "chat_history": []}}
response = requests.post("http://localhost:8000/invoke", json=inputs)

print(response.json()['output'])
```
output

```bash
De acordo com o texto fornecido, um plano de disciplina deve incluir os seguintes elementos:
I - identificação;
II - ementa;
III - objetivos: geral e específicos;
IV - conteúdo programático;
V - metodologia;
VI - recursos;
VII - avaliação; e
VIII - referências (básica e complementar).

```


#### **Resposta Esperada**
A API retornará uma resposta gerada pelo pipeline RAG, usando os documentos carregados como base de conhecimento.

---

### **Passo 4: Usar a API em uma Aplicação**
Agora que o chatbot está disponível como uma API REST, você pode integrá-lo a:
1. Aplicações Web (usando frameworks como **React** ou **Flask**).
2. Aplicações Móveis (Android ou iOS).
3. Interfaces de Linha de Comando que consomem a API.

---

### **Passo 5: Expansões Finais**
1. **Autenticação**: Adicione autenticação para proteger sua API (ex: tokens de autenticação).
2. **Deploy em Nuvem**:
   - Use plataformas como **Heroku**, **AWS**, ou **Azure** para disponibilizar a API globalmente.
3. **Monitoramento**: Adicione logs para monitorar o uso da API.

Se precisar de ajuda com o deploy ou integração em um front-end, é só chamar!