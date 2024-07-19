import os
import openai
import logging
import sys
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

app = Flask(__name__)

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize OpenAI API from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Function to extract content from PDF using PyPDF2
def extract_content_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        content = ""
        for page in reader.pages:
            content += page.extract_text()
    return content

# Save extracted content to a text file
def save_content_to_text_file(content, text_file_path):
    with open(text_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Initialize the index
pdf_path = "data/Questions.pdf"
text_file_path = "data/questions.txt"

# Extract and save content from PDF
content = extract_content_from_pdf(pdf_path)
save_content_to_text_file(content, text_file_path)

# Check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # Load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Create a query engine
query_engine = index.as_query_engine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    relevant_content = query_engine.query(user_query)
    response = get_openai_response(user_query, relevant_content)
    return jsonify({'response': response})

def get_openai_response(question, additional_info):
    messages = [
        {"role": "system", "content": "You are a virtual assistant, representing the plants store, Little Shoots. "},
        {"role": "user", "content": f"Here is some additional information that might help answer the question: {additional_info}\n\nAnswer the following question clearly and accurately: {question}"}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=350
        )
        logging.info(f"OpenAI response: {response}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error communicating with OpenAI: {e}")
        return "Sorry, there was an error processing your request."

if __name__ == "__main__":
    app.run(debug=True)
