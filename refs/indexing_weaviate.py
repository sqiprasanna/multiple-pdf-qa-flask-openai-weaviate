import weaviate
import fitz  # PyMuPDF
import os
from weaviate.auth import AuthApiKey
from weaviate.util import generate_uuid5
import weaviate.classes as wvc
from weaviate.classes.init import AdditionalConfig, Timeout
import openai

# Set your OpenAI API key
openai.api_key = ""  # Replace with your actual OpenAI API key

# Initialize Weaviate client with authentication
client = weaviate.connect_to_wcs(
    cluster_url="https://yeezwn30t6yvvrpdtsd6eg.c0.us-east1.gcp.weaviate.cloud",
    auth_credentials=AuthApiKey(api_key=""),  # Replace with your actual API key
    headers={
        "X-OpenAI-Api-Key": openai.api_key  # Use the same OpenAI API key
    },
    additional_config=AdditionalConfig(
        timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
    ),
    skip_init_checks=True
)

try:
    # Check if the class already exists
    try:
        collections = client.collections.get("Document")
    except Exception as e:
        collections = client.collections.create("Document",
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            properties=[
                wvc.config.Property(
                    name="content",
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=True,  # Include the property name ("content") when vectorizing
                    tokenization=wvc.config.Tokenization.LOWERCASE  # Use "lowercase" tokenization
                ),
             ]
        )

    def overlap_chunking(text, chunk_size=2000, overlap_size=200):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap_size):
            chunk = words[i:i + chunk_size]
            if chunk:
                chunks.append(" ".join(chunk))
        return chunks

    # Function to extract text from a PDF
    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    # Function to create embedding using OpenAI API
    def create_embedding(text):
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"  # Specify the embedding model to use
        )
        embedding = response["data"][0]["embedding"]
        return embedding

    # Function to index PDF content into Weaviate
    def index_pdf(pdf_path):
        text = extract_text_from_pdf(pdf_path)
        chunks = overlap_chunking(text)
        data_rows = [{"content": chunk, "vector": create_embedding(chunk)} for chunk in chunks]
        
        collection = client.collections.get("PDFDocument")
        
        with collection.batch.dynamic() as batch:
            for data_row in data_rows:
                obj_uuid = generate_uuid5(data_row["content"])
                batch.add_object(
                    properties={"content": data_row["content"]},
                    vector=data_row["vector"],
                    uuid=obj_uuid
                )

    # Index all PDF files in the static folder
    pdf_folder = "files"
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            index_pdf(os.path.join(pdf_folder, pdf_file))

finally:
    client.close()
