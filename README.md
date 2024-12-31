Abaixo, apresento um guia passo a passo para criar um chatbot com **RAG** (Retrieval-Augmented Generation) usando **LangChain**, **OpenAI**, e **Groq**. 

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

