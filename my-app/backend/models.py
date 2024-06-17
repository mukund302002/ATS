from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import httpx
import logging
import fitz
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
def scoring(job_description,content,id):
    return


# def row_count(name):
#     table_name = name
#     logging.debug(f"table name is {table_name}")
#     if not table_name:
#         return jsonify({'status': 'failure', 'message': 'Table name is required'}), 400

#     try:
#         # Fetch the row count
#         response = supabase_client.table(table_name).select("*", count="exact").execute()
#         if response.count is not None:
#             logging.debug(f"response fetched of row count {response.count}")
#             row_count = response.count
#             return row_count
#         else:
#             return jsonify({'status': 'failure', 'message': 'Failed to fetch row count'})
#     except Exception as e:
#         logging.debug(f"Failed to fetch row count: {e}")
#         return jsonify({'status': 'failure', 'message': 'Failed to fetch row count'})
    


def extract_pdf_content(id,category):
    data = request.json
    

    # Fetch the row with the URL and ID from Supabase
    response = supabase_client.table(category).select('url').eq('id', id).execute()
    if response.data:
        logging.debug("URL and ID fetched successfully")
        pdf_url = response.data[0]['url']

        # Download and process the PDF (same as before)
        pdf_response = httpx.get(pdf_url)
        pdf_response.raise_for_status()
        pdf_content = pdf_response.content

        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        text_content = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_content += page.get_text()

        return text_content
    else:
        logging.debug("No data found for the given category")
        return jsonify({'status': 'failure', 'message': 'No data found'})



def compute_score(content,description):
    return 1