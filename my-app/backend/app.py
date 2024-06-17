from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import httpx
import logging
import fitz
from models import extract_pdf_content,compute_score


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





def test_connection():
    try:
        response = httpx.get(SUPABASE_URL)
    except Exception as e:
        logging.debug(f"failed ot connect to database")
        
@app.route('/api/prompt', methods=['POST'])
def prompt():
    data = request.json
    query = data.get('prompt')
    category=data.get('category')
    number = data.get('numCandidates')


    response = supabase_client.table('Job Description').select('JD').eq('category', category).execute()

    if response.data:
        logging.debug(f"jd fetched successfully")
        
    job_description=response.data[0]['JD']
    logging.debug(f"jd{job_description}")

    # num_rows=row_count(category)
    id=1
    # for id in range(num_rows):
    # response = supabase_client.table(category).select('id').eq('category', category).execute()
    response = supabase_client.table(category).select('url').eq('id', id).execute()
    pdf_data=""
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
            pdf_data=text_content

        # return text_content
    else:
        logging.debug("No data found for the given category")
        return jsonify({'status': 'failure', 'message': 'No data found'})
    

    # pdf_content=extract_pdf_content(id,category)
    logging.debug("pdf content {pdf_data}")

    score=compute_score(pdf_content,job_description)
        
    update_score = supabase_client.table(category).update({'score': score}).eq('id', id).execute()
    if update_score.data:
        logging.debug("Score updated successfully")
        return jsonify({'status': 'success', 'message': 'Score updated successfully'})
    else:
        logging.debug("Failed to update score")
        return jsonify({'status': 'failure', 'message': 'Failed to update score'})
        













if __name__ == '__main__':
    app.run(debug=True, port=5000)
