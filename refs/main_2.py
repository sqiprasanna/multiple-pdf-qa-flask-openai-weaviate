from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv, find_dotenv
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
import time
import requests
from requests.packages.urllib3.util.ssl_ import create_urllib3_context

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
INDEX_NAME = os.getenv('INDEX_NAME')

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine')
index = pinecone.Index(INDEX_NAME)

# Ciphers setup
CIPHERS = (
    'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:ECDH+AESGCM:ECDH+CHACHA20:DH+AESGCM:DH+CHACHA20:'
    'ECDHE+AES:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4:!HMAC_SHA1:!SHA1:!DHE+AES:!ECDH+AES:!DH+AES'
)

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = CIPHERS
requests.packages.urllib3.util.ssl_.create_default_context = create_urllib3_context

# Wait for the index to be fully initialized
time.sleep(20)

index_stats = index.describe_index_stats()
print(f"Completed ciphers load, GRPCIndex index_stats: \n {index_stats}")

# Initialize embeddings
embed = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=OPENAI_API_KEY
)
print("OpenAI embeddings completed")

# Function to extract and chunk text from PDFs
def extract_and_chunk_pdf(file_path):
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    
    return texts

# Index documents from a specified directory
def index_documents_from_directory(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(directory, file_name)
            texts = extract_and_chunk_pdf(file_path)

            for i, text in enumerate(texts):
                embedding = embed.embed_query(text)
                index.upsert([(f"{file_name}_{i}", embedding, {'text': text})])
    print("Documents indexed successfully")

# Call the function to index documents
index_documents_from_directory("./files")

# Initialize vector store
text_field = "text"
vectorstore = Pinecone(index, embed.embed_query, text_field)
print("Vector store initialized")

# Initialize chat model
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
print(f"Initialized OpenAI chat model LLM model name 'gpt-3.5-turbo', temperature 0.0")

# Initialize QA handlers
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ask/")
async def ask_question(query: str):
    try:
        answer = qa.run(query)
        answer_with_sources = qa_with_sources(query)
        return {
            "answer": answer,
            "answer_with_sources": answer_with_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
def search(query: str):
    result = f"You searched for: {query}"
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
