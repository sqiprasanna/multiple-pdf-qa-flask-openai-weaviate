from flask import Flask, request, jsonify, render_template
import weaviate
import openai

app = Flask(__name__)
app.secret_key = '12345678'  # Needed for session management

openai_key = "your-openapi-key"  # Replace with your actual OpenAI API key
lm_client = openai.OpenAI(api_key=openai_key)

# Initialize Weaviate client with API key authentication
client = weaviate.Client(
    url="https://yeezwn30t6yvvrpdtsd6eg.c0.us-east1.gcp.weaviate.cloud",
    auth_client_secret=weaviate.auth.AuthApiKey(api_key="weaviate-key"),
    additional_headers={
        "X-OpenAI-Api-Key": openai_key,
    }
)

def search_documents(query):
    try:
        # Search for chunks that match the query embedding
        result_embedding = client.query.get("Document", ["text", "metadata"]).with_near_text({
            "concepts": [query]
        }).with_limit(10).do()
        embedding_chunks = result_embedding['data']['Get']['Document'] if 'data' in result_embedding and 'Get' in result_embedding['data'] and 'Document' in result_embedding['data']['Get'] else []

        # Search as before for relevant chunks
        result_text = client.query.get("Document", ["text", "metadata"]).with_near_text({
            "concepts": [query]
        }).with_limit(10).do()
        text_chunks = result_text['data']['Get']['Document'] if 'data' in result_text and 'Get' in result_text['data'] and 'Document' in result_text['data']['Get'] else []

        # Combine both results
        combined_chunks = embedding_chunks + text_chunks
        return combined_chunks
    except Exception as e:
        print(f"Error querying Weaviate: {e}")
        return []

def generate_pdf_qa_response(query):
    context = search_documents(query)
    print(context)
    if not context:
        return "No relevant information found in the database."

    context_texts = "\n\n".join([f"Document: {doc['metadata']}\n\n{doc['text']}" for doc in context])
    system_message = "You are provided with paragraphs from documents based on your query. Use the information to answer the query effectively."
    user_message = f"Query: {query}\nContext: {context_texts}\nAnswer:"

    print("--------------------- USER MESSAGE AND RETRIEVED CONTEXT:- \n ------------------", user_message)

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    try:
        response = lm_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=250,
            temperature=0.5
        )
        print(response)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error while generating response from GPT: {e}")
        return "Failed to generate a response."

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.form['query']
    response = generate_pdf_qa_response(query)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
