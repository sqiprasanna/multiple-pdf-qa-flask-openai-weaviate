from flask import Flask, request, jsonify
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig, Timeout
import weaviate.classes as wvc
from weaviate.classes.query import MetadataQuery

app = Flask(__name__)

# Initialize Weaviate client with authentication
client = weaviate.connect_to_wcs(
    cluster_url="https://yeezwn30t6yvvrpdtsd6eg.c0.us-east1.gcp.weaviate.cloud",
    auth_credentials=AuthApiKey(api_key=""),  # Replace with your actual API key
    headers={
        "X-OpenAI-Api-Key": ""  # If using OpenAI module, replace with your actual OpenAI API key
    },
    additional_config=AdditionalConfig(
        timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
    ),
    skip_init_checks=True
)

@app.route("/index_pdfs", methods=["POST"])
def index_pdfs():
    from index_pdfs import index_all_pdfs
    index_all_pdfs()
    return jsonify({"message": "PDFs indexed successfully"}), 200

@app.route("/query", methods=["POST"])
def query_pdf():
    query = request.json.get("query")
    pdf_document = client.collections.get("PDFDocument")
    print("Query:", query)
    # Create embedding for the query using OpenAI API
    embedding_response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"  # Specify the embedding model to use
    )
    embedding = embedding_response["data"][0]["embedding"]

    # Perform the near_vector query
    response = pdf_document.query.near_vector(
        vector=embedding,
        limit=3,
        return_metadata=MetadataQuery(distance=True)
    ).do()

    results = []
    for obj in response.objects:
        results.append({
            "properties": obj.properties,
            "metadata": {"distance": obj.metadata.distance}
        })
    
    return jsonify(results)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    client.close()
    return 'Client closed', 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
