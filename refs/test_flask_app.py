import requests
import json

BASE_URL = "http://localhost:5000"

def test_index_pdfs():
    response = requests.post(f"{BASE_URL}/index_pdfs")
    print("Index PDFs Response:", response.status_code, response.json())


def test_query_pdf(query):
    payload = {"query": query}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(f"{BASE_URL}/query", data=json.dumps(payload), headers=headers)
    print("Query PDFs Response:", response.status_code, response.json())

def test_shutdown():
    response = requests.post(f"{BASE_URL}/shutdown")
    print("Shutdown Response:", response.status_code, response.text)

if __name__ == "__main__":
    # Test indexing PDFs
    # test_index_pdfs()

    # Test querying the indexed PDFs
    test_query_pdf("Whos is Sai Prasanna Kumar?")

    # Test shutting down the Flask application
    # test_shutdown()
