from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text


path_to_pdfs = "./files"
pdf_texts = {}
for pdf_file in os.listdir(path_to_pdfs):
    if pdf_file.endswith('.pdf'):
        pdf_texts[pdf_file] = extract_text_from_pdf(os.path.join(path_to_pdfs, pdf_file))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def chunk_text_overlap(text, chunk_size= 1000, overlap= 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = overlap)
    chunks = splitter.split_text(text)
    return chunks

chunked_texts = {}
for pdf_name, pdf_text in pdf_texts.items():
    chunked_texts[pdf_name] = chunk_text_overlap(pdf_text)


####### PINE CONE - connect and push the chunks
pc = Pinecone(api_key="")
index_name = "chatpdf"
print(pc.list_indexes())


index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(openai_api_key = "")
for pdf_name, chunks in chunked_texts.items():
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk)
        index.upsert([(f"{pdf_name}_{i}",embedding, {'text':chunk})])

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)
print("vector store initialized")

# Initialize chat model
llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
print(f"Initialized openai chat model LLM model name {model_name}, temperature 0.0," )

class Query(BaseModel):
    question: str

# Initialize QA handlers
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())


@app.post("/query/")
async def query_pdf(query:Query):
    try:
        answer = qa.run(query)
        answer_with_sources = qa_with_sources(query)
        return {
            "answer": answer,
            "answer_with_sources": answer_with_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

