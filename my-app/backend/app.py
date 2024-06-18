# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from supabase import create_client, Client
# from sqlalchemy import create_engine
# from dotenv import load_dotenv
# import os
# import httpx
# import logging
# import fitz
# import requests
# from models import process_pdfs,compute_similarity,extract_text_from_pdf,fetch_urls
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# load_dotenv()

# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)
# CORS(app) 
# DATABASE_URL = os.getenv('DATABASE_URL')
# engine = create_engine(DATABASE_URL)
# SUPABASE_URL = os.getenv('SUPABASE_URL')
# SUPABASE_KEY = os.getenv('SUPABASE_KEY')
# print(SUPABASE_KEY)
# supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)





# def test_connection():
#     try:
#         response = httpx.get(SUPABASE_URL)
#     except Exception as e:
#         logging.debug(f"failed ot connect to database")

# @app.route('/')

# def fetch_pdf_urls_from_bucket(bucket_name):
#     response = supabase_client.storage.from_(bucket_name).list()
#     pdf_urls = [f"https://your-supabase-url.supabase.co/storage/v1/object/public/{bucket_name}/{file['name']}" for file in response if file['name'].endswith('.pdf')]
#     logging.debug(f"fetched pdf urls{pdf_urls}")
#     return pdf_urls

# # Function to get the current max ID from a table
# # def get_max_id(table_name):
# #     response = supabase_client.table(table_name).select('id').order('id', desc=True).limit(1).execute()
# #     if response.data:
# #         return response.data[0]['id']
# #     return 0

# # Function to store PDF URLs in the respective table
# def store_pdf_urls_in_table(table_name, pdf_urls):
#     # max_id = get_max_id(table_name)
#     for  pdf_url in enumerate(pdf_urls):
#         response = supabase_client.table(table_name).insert({
           
#             'url': pdf_url
#         }).execute()
#         if response.status_code != 201:
#             logging.error(f"Failed to insert URL {pdf_url} into table {table_name}: {response}")
#         else:
#             logging.debug(f"Inserted URL {pdf_url} into table {table_name} ")

# # Define the buckets and their respective tables
# buckets_tables = {
#     'machine learning': 'machine learning',
#     'software developer': 'software developer',
#     'project manager': 'project manager'
# }

# logging.info("Starting the PDF URL extraction and storage process...")
# for bucket_name, table_name in buckets_tables.items():
#     pdf_urls = fetch_pdf_urls_from_bucket(bucket_name)
#     store_pdf_urls_in_table(table_name, pdf_urls)
#     logging.info(f"Processed bucket {bucket_name} and stored URLs in table {table_name}")
        
# @app.route('/api/prompt', methods=['POST'])
# def prompt():
#     data = request.json
#     query = data.get('prompt')
#     category=data.get('category')
#     number = data.get('numCandidates')

#     def fetch_pdf_urls_from_bucket(bucket_name):
#         response = supabase_client.storage.from_(bucket_name).list()
#         pdf_urls = [f"https://your-supabase-url.supabase.co/storage/v1/object/public/{bucket_name}/{file['name']}" for file in response if file['name'].endswith('.pdf')]
#         logging.debug(f"urls fetched success{pdf_urls}")
#         return pdf_urls
    
#     def get_max_id(table_name):
#         response = supabase_client.table(table_name).select('id').order('id', desc=True).limit(1).execute()
#         if response.data:
#             return response.data[0]['id']
#         logging.debug("got the max id")
#         return 0
    
#     def store_pdf_urls_in_table(table_name, pdf_urls):
#         max_id = get_max_id(table_name)
#         for idx, pdf_url in enumerate(pdf_urls, start=max_id + 1):
#             response = supabase_client.table(table_name).insert({
#                 'id': idx,
#                 'url': pdf_url
#             }).execute()
#             if response.status_code != 201:
#                 logging.error(f"Failed to insert URL {pdf_url} into table {table_name}: {response}")
#             else:
#                 logging.debug(f"Inserted URL {pdf_url} into table {table_name} with id {idx}")

#     # buckets_tables = {
#     #     'machine learning': 'machine learning',
#     #     'software development': 'software development',
#     #     'project manager': 'project manager'
#     # }

#     # for bucket_name, table_name in buckets_tables.items():
#     pdf_urls = fetch_pdf_urls_from_bucket(category)
#     store_pdf_urls_in_table(category, pdf_urls)
#     logging.info(f"Processed bucket {category} and stored URLs in table {category}")


#     response = supabase_client.table('Job Description').select('JD').eq('category', category).execute()

#     if response.data:
#         logging.debug(f"jd fetched successfully")
        
#     job_description=response.data[0]['JD']
#     logging.debug(f"jd{job_description}")

#     def process_pdfs(job_description,category):
#         logging.debug(f"process pdfs started")
#         urls = fetch_urls(category)
#         logging.debug(f"urls fetched successfully{urls}")
#         for entry in urls:
#             pdf_text = extract_text_from_pdf(entry['url'])
#             score = compute_similarity(pdf_text, job_description)
#             # Update the database with the similarity score
#             supabase_client.table(category).update({'score': score}).eq('id', entry['id']).execute()
#             logging.debug(f"updated score to database successfully")



#     process_pdfs(job_description,category)



#     return
        













# if __name__ == '__main__':
#     app.run(debug=True, port=5000)






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
from models import process_pdfs, compute_similarity, extract_text_from_pdf, fetch_urls
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def fetch_pdf_urls_from_bucket(bucket_name):
    try:
        response = supabase_client.storage.from_(bucket_name).list()
        if not response:
            logging.warning(f"No files found in bucket {bucket_name}")
            return []
        pdf_urls = [supabase_client.storage.from_(bucket_name).get_public_url(file['name'])['publicURL'] for file in response if file['name'].endswith('.pdf')]
        logging.debug(f"Fetched PDF URLs from bucket {bucket_name}: {pdf_urls}")
        return pdf_urls
    except Exception as e:
        logging.error(f"Exception while fetching from bucket {bucket_name}: {e}")
        return []




def url_exists_in_table(table_name, url):
    response = supabase_client.table(table_name).select('url').eq('url', url).execute()
    return bool(response.data)

def store_pdf_urls_in_table(table_name, pdf_urls):
    for pdf_url in pdf_urls:
        if not url_exists_in_table(table_name, pdf_url):
            response = supabase_client.table(table_name).insert({'url': pdf_url}).execute()
            if response.status_code != 201:
                logging.error(f"Failed to insert URL {pdf_url} into table {table_name}: {response}")
            else:
                logging.debug(f"Inserted URL {pdf_url} into table {table_name}")
        else:
            logging.debug(f"URL {pdf_url} already exists in table {table_name}")

def initialize_tables_with_pdf_urls():
    buckets_tables = {
        'machine learning': 'machine learning',
        'software-developer': 'software-developer',
        'project manager': 'project manager'
    }

    for bucket_name, table_name in buckets_tables.items():
        pdf_urls = fetch_pdf_urls_from_bucket(bucket_name)
        if pdf_urls:
            store_pdf_urls_in_table(table_name, pdf_urls)
            logging.info(f"Processed bucket {bucket_name} and stored URLs in table {table_name}")
        else:
            logging.warning(f"No URLs found in bucket {bucket_name}")

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
        for entry in urls:
            pdf_text = extract_text_from_pdf(entry['url'])
            score = compute_similarity(pdf_text, job_description)
            supabase_client.table(category).update({'score': score}).eq('id', entry['id']).execute()
            logging.debug("Updated score in database successfully")

    process_pdfs(job_description, category)

    return jsonify({"status": "success", "message": "Processed PDF URLs and updated scores"}), 200

if __name__ == '__main__':
    initialize_tables_with_pdf_urls()
    app.run(debug=True, port=5000)

