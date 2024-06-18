from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import httpx
import logging
import fitz
import requests
from models import process_pdfs, compute_similarity, extract_text_from_pdf, fetch_urls,get_embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv()

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
print(SUPABASE_KEY)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# def fetch_pdf_urls_from_bucket(bucket_name):
#     try:
#         response = supabase_client.storage.from_(bucket_name).list()
#         if not response:
#             logging.warning(f"No files found in bucket {bucket_name}")
#             return []
#         pdf_urls = [supabase_client.storage.from_(bucket_name).get_public_url(file['name'])['publicURL'] for file in response if file['name'].endswith('.pdf')]
#         logging.debug(f"Fetched PDF URLs from bucket {bucket_name}: {pdf_urls}")
#         return pdf_urls
#     except Exception as e:
#         logging.error(f"Exception while fetching from bucket {bucket_name}: {e}")
#         return []




# def url_exists_in_table(table_name, url):
#     response = supabase_client.table(table_name).select('url').eq('url', url).execute()
#     return bool(response.data)

# def store_pdf_urls_in_table(table_name, pdf_urls):
#     for pdf_url in pdf_urls:
#         if not url_exists_in_table(table_name, pdf_url):
#             response = supabase_client.table(table_name).insert({'url': pdf_url}).execute()
#             if response.status_code != 201:
#                 logging.error(f"Failed to insert URL {pdf_url} into table {table_name}: {response}")
#             else:
#                 logging.debug(f"Inserted URL {pdf_url} into table {table_name}")
#         else:
#             logging.debug(f"URL {pdf_url} already exists in table {table_name}")

# def initialize_tables_with_pdf_urls():
#     buckets_tables = {
#         'machine learning': 'machine learning',
#         'software-developer': 'software-developer',
#         'project manager': 'project manager'
#     }

#     for bucket_name, table_name in buckets_tables.items():
#         pdf_urls = fetch_pdf_urls_from_bucket(bucket_name)
#         if pdf_urls:
#             store_pdf_urls_in_table(table_name, pdf_urls)
#             logging.info(f"Processed bucket {bucket_name} and stored URLs in table {table_name}")
#         else:
#             logging.warning(f"No URLs found in bucket {bucket_name}")

@app.route('/api/prompt', methods=['POST'])
def prompt():
    data = request.json
    query = data.get('prompt')
    category = data.get('category')
    number = data.get('numCandidates')

    response = supabase_client.table('Job Description').select('JD').eq('category', category).execute()

    if response.data:
        logging.debug("Job description fetched successfully")
        
    job_description = response.data[0]['JD']
    logging.debug(f"Job Description: {job_description}")

    def process_pdfs(job_description, category):
        logging.debug("Process PDFs started")
        urls = fetch_urls(category)
        logging.debug(f"URLs fetched successfully: {urls}")
        logging.debug(f"datatype of urls: {type(urls)}")
        size=len(urls)
        logging.debug(f"list has {size} urls")
        for entry in range(size):

            pdf_text = extract_text_from_pdf(urls[entry])
            #storing the score in the column
            logging.debug("going to enter the compute similarity function")
            score = compute_similarity(pdf_text, job_description,model,util)

            logging.debug(f"computed the score {score}")
            # supabase_client.table(category).update({'score': score}).eq('url', urls[entry]).execute()
            logging.debug(f"updated the score column")

            #storing the embeddings in the column
            embeddings = get_embeddings(pdf_text)
            supabase_client.table(category).update({'score': score, 'embeddings': embeddings}).eq('url', urls[entry]).execute()

            logging.debug(f"updated the embeddings column")
            logging.debug("Updated score and embeddings in database successfully")
        return      
    #processing the pdf by updating the score and embeddings to the table
    process_pdfs(job_description, category)

    #process query
    def process_query(query):
        response = supabase_client.table(category).select('url').order('score', desc=True).execute()
        logging.debug(f"debugging the response {response.data}")
        urls = [row['url'] for row in response.data]
        logging.debug(f"fetched the content in decending order {urls}")

        embeddings=model.encode(query,convert_to_tensor=True)
        embeddings_list=embeddings.tolist()

        return
    process_query(query)

    return jsonify({"status": "success", "message": "Processed PDF URLs and updated scores"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)

