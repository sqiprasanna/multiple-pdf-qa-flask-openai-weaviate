import weaviate
import fitz  # PyMuPDF
import os
import glob
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
openai_key = ""  # Replace with your actual OpenAI API key

# Initialize Weaviate client with API key authentication
client = weaviate.Client(
    url="https://yeezwn30t6yvvrpdtsd6eg.c0.us-east1.gcp.weaviate.cloud",
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=""),
    additional_headers={
        "X-OpenAI-Api-Key": openai_key,
    }
)

def split_pdf_text_by_page(pdf_path):
    pages = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text() if page.extract_text() else ""
            pages.append(text)
    return pages

def tokenize_text(text):
    words = nltk.word_tokenize(text)
    return words

def chunk_text(words, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)
    return chunks

def generate_tags(text, n_tags=5):
    vectorizer = TfidfVectorizer(max_features=n_tags)
    X = vectorizer.fit_transform([text])
    tags = vectorizer.get_feature_names_out()
    return tags

def load_documents(directory, glob_patterns):
    documents = []
    for pattern in glob_patterns:
        for path in glob.glob(os.path.join(directory, pattern)):
            try:
                if path.endswith('.docx'):
                    text = docx2txt.process(path)
                    documents.extend([(text, os.path.basename(path), 1)])  # Single page docx
                elif path.endswith('.pdf'):
                    pages = split_pdf_text_by_page(path)
                    for page_text in pages:
                        words = tokenize_text(page_text)
                        chunks = chunk_text(words)
                        for chunk in chunks:
                            tags = generate_tags(chunk)
                            documents.append((chunk, os.path.basename(path), tags))
            except Exception as e:
                print(f"Warning: The file {path} could not be processed. Error: {e}")
    return documents

def index_documents(documents, class_name):
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for text, filename, tags in documents:
            batch.add_data_object({
                "text": text,
                "metadata": f"{filename} - Tags: {', '.join(tags)}"
            }, class_name)
    
def setup_weaviate_schema(class_name):
    schema = {
        "class": class_name,
        "properties": [
            {"name": "text", "dataType": ["text"], "indexInverted": True},
            {"name": "metadata", "dataType": ["text"]}
        ],
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "vectorizeClassName": False,
                "model": "ada",
                "modelVersion": "002",
                "type": "text"
            }
        }
    }
    client.schema.delete_class(class_name)  # Clean slate
    client.schema.create_class(schema)

if __name__ == "__main__":
    directory = "static"  # Your directory with PDF files
    class_name = "Document"
    glob_patterns = ["*.docx", "*.pdf"]

    setup_weaviate_schema(class_name)
    documents = load_documents(directory, glob_patterns)
    index_documents(documents, class_name)
