import os
import streamlit as st
# from dotenv import load_dotenv
import logging, requests
from sentence_transformers import util
import google.generativeai as genai
from io import BytesIO
import numpy as np
import json, httpx, time
from tenacity import retry, stop_after_attempt, wait_fixed
import PyPDF2
from supabase import create_client, Client
from sqlalchemy import create_engine

# Load environment variables
DATABASE_URL = st.secrets["DATABASE_URL"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
API_KEY_GEMINI = st.secrets["API_TOKEN_GEMINI"]

# Set up logging
logging.basicConfig(level=logging.INFO)

# Connect to the database
engine = create_engine(DATABASE_URL)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize session state
if 'context' not in st.session_state:
    st.session_state.context = "I'll provide you Selected resume content and User query, you have to reply the query according to this selected resume content.\n\n"
if 'prompt' not in st.session_state:
    st.session_state.prompt = "You are now a chatbot. You will get the initial context and then pairs of User: and Assistant:. So you have to answer keeping the context and previous chats and give answer like a chatbot."
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = True

# Function to extract text from PDF file
def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

def make_embed_text_fn(model):
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def embed_fn(text: str) -> list[float]:
        embedding = genai.embed_content(model=model, content=text, task_type="classification")
        return embedding['embedding']
    return embed_fn

# Function to get embeddings of text using Gemini embeddings
def get_embeddings(text):
    genai.configure(api_key=API_KEY_GEMINI)
    model = 'models/embedding-001'
    try:
        embed_fn = make_embed_text_fn(model)
        embeddings = embed_fn(text)
        return embeddings
    except Exception as e:
        logging.error(f"Error getting embeddings: {e}")
        return []

# Function to create input text for LLM
def create_input_text(all_pdf_texts, number, query_text):
    input_text = "You are a hiring manager at a company. You have received multiple resumes for a job opening regarding the job description. Now you have to answer the Query with these documents I am giving to you(give very precise and short answer and give only what is asked in a summarized way):\n\n"
    input_text += f"Query:\n{query_text}\n"
    input_text += f"Number of candidates you have to output:\n{number}\n"
    for i, pdf_text in enumerate(all_pdf_texts, 1):
        input_text += f"Document {i}:\n{pdf_text}\n\n"
    return input_text

def get_chat_response_from_llm(chat_input_text):
    genai.configure(api_key=API_KEY_GEMINI)
    model = genai.GenerativeModel('gemini-1.5-flash')
    output = model.generate_content(chat_input_text)
    response = output.text
    return response

# Function to get response from LLM
def get_response_from_llm(input_text):
    genai.configure(api_key=API_KEY_GEMINI)
    model = genai.GenerativeModel('gemini-1.5-flash')
    output = model.generate_content(input_text)
    response = output.text
    return response

def process_file(uploaded_file, job_description_embeddings):
    try:
        file_content = uploaded_file.read()
        content = extract_text_from_pdf(BytesIO(file_content))
        if content:
            content_embeddings = get_embeddings(content)
            score = util.cos_sim(job_description_embeddings, content_embeddings)
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    supabase_client.table('resumes').insert({
                        'resumetext': content,
                        'score': score.item(),
                        'embedding': content_embeddings
                    }).execute()
                    break
                except httpx.HTTPStatusError as e:
                    st.warning(f"HTTP error occurred: {e.response.text}")
                    retry_count += 1
                    time.sleep(1)
                except Exception as e:
                    st.warning(f"An error occurred: {str(e)}")
                    retry_count += 1
                    time.sleep(1)
        else:
            st.warning(f"Empty content extracted from PDF: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

# Streamlit application
st.title("ATS CHATBOT")

# Custom CSS to hide the sidebar
if not st.session_state.sidebar_expanded:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

# files uploaded checker
if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = False

# Step 1: Upload files and job description
if st.session_state.sidebar_expanded:
    st.sidebar.header("Upload Resumes and Job Description")
    job_desc = st.sidebar.text_area("Job Description")
    uploaded_files = st.sidebar.file_uploader("Upload Resume Files", type=["pdf"], accept_multiple_files=True)

    if st.sidebar.button("Submit Initial Data"):
        with st.spinner("Processing..."):
            supabase_client.table('resumes').delete().neq("id", 0).execute()
            job_description_text = "You are a hiring manager at a company. You work is to judge the resumes on the basis of job description. You have to figure out some key points from the job description which I can easily check in the candidates' resumes. I'll give you the job description. These are the key points I needed from the job description: 1. Qualifications required for the job. 2. Skills required for this job. 3. Preferred skills 4. Candidates roles and responsibilities"
            job_description_text += f"\n\nJob Description:\n{job_desc}"
            job_description_response = get_response_from_llm(job_description_text)
            job_description_embeddings = get_embeddings(job_description_response)
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    process_file(uploaded_file, job_description_embeddings)
                    time.sleep(1)
                    st.session_state.files_uploaded = True
                st.session_state.sidebar_expanded = False
                st.sidebar.success("Uploaded files and job description successfully")
                # st.session_state.sidebar_expanded = False
                # st.success("Now you can submit a prompt and number of candidates to shortlist")

# Step 2: Prompt input
number = st.number_input("Enter the number of candidates to shortlist", min_value=1, step=1,key='number')

# st.header("Submit a Prompt")
query = st.text_input("Enter your First prompt", key='query')
# number = st.number_input("Enter the number of candidates to shortlist", min_value=1, step=1)

if 'first_response' not in st.session_state:
    st.session_state.first_response = False

if st.button("Answer"):
    if not st.session_state.files_uploaded:
        st.warning("Please upload files and job description first")
    else:
        with st.spinner("Processing..."):
            response = supabase_client.table('resumes').select('resumetext', 'embedding', 'score').order('score', desc=True).limit(100).execute()
            if query and number:
                resume_content = [row['resumetext'] for row in response.data]
                top_n_embeddings = [row['embedding'] for row in response.data]
                resume_scores = [row['score'] for row in response.data]
                resume_score_100 = [score * 100.0 for score in resume_scores]
                top_n_embeddings_list = [json.loads(embedding) for embedding in top_n_embeddings]
                query_embedding = get_embeddings(query)
                if len(query_embedding) == 0:
                    st.error("Empty query embedding received")
                try:
                    top_n_embeddings_np = [np.array(embedding, dtype=float) for embedding in top_n_embeddings_list]
                    query_embedding_np = np.array(query_embedding, dtype=float)
                except ValueError as e:
                    st.error(f"Error converting embeddings to numpy arrays: {e}")
                    st.write({'error': 'Invalid embeddings format'}), 500
                similarities = []
                for embeddings in top_n_embeddings_np:
                    similarity = util.cos_sim(query_embedding_np, embeddings)
                    similarities.append(similarity.item())
                similarities = [score * 100.0 for score in similarities]
                updated_similarity_score = [a + (b * 1.5) for a, b in zip(similarities, resume_score_100)]
                similarities = np.array(updated_similarity_score)
                similarity_position = np.argsort(similarities)
                top_indices = sorted(range(len(similarity_position)), key=lambda i: similarity_position[i], reverse=True)
                output_indices = top_indices[:int(number)]
                all_pdf_texts = [resume_content[idx] for idx in output_indices]
                all_pdf_texts_str = "\n\n".join(all_pdf_texts)
                st.session_state.context += f"\nSelected resumes content:\n{all_pdf_texts_str}"
                st.session_state.context += f"\n\n{query}\n"
                input_text = create_input_text(all_pdf_texts, number, query)
                response = get_chat_response_from_llm(input_text)
                st.session_state.prompt += f"User: {query}\nAssistant: {response}\n"
                st.session_state.first_response = True
                # st.success("Prompt processed successfully")
                
                # Append the query and response to the session state messages
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.messages.append({"role": "assistant", "content": response})
def display_chat_message(role, content):
    if role == "user":
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{content}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)

# Display the chat history
if st.session_state.first_response:
    # st.header("Chat History")
    
    # Display the first query and response separately
    if len(st.session_state.messages) >= 2:
        first_query = st.session_state.messages[0]
        first_response = st.session_state.messages[1]
    
    display_chat_message(first_query["role"], first_query["content"])
    display_chat_message(first_response["role"], first_response["content"])

# Display the rest of the conversation
for message in st.session_state.messages[2:]:
    display_chat_message(message["role"], message["content"])

# Handle new user input
if user_input := st.chat_input("You:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    display_chat_message("user", user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            message_placeholder = st.empty()
            input_text = f"{st.session_state.context}\n\n{st.session_state.prompt}\nUser: {user_input}\nAssistant: "
            response = get_chat_response_from_llm(input_text)
            message_placeholder.markdown(f'<div class="assistant-message">{response}</div>', unsafe_allow_html=True)
    
    st.session_state.prompt += f"User: {user_input}\nAssistant: {response}\n"
    st.session_state.messages.append({"role": "assistant", "content": response})


st.markdown("""
<style>
    .message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .user-message, .assistant-message {
        max-width: 70%;
        padding: 10px;
        border-radius: 10px;
    }
    .user-message {
        background-color: #DCF8C6;
        color: #000;
        align-self: flex-end;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #F1F0F0;
        color: #000;
        align-self: flex-start;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)
