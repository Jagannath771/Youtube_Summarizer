import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from claims.scraper import PubMedScraper
from claims.doc_loader import load_documents
from claims.utils import ranked_df
from claims.retrieval import InMemoryVectorStore, CustomRetriever
from claims.rag import RAGQueryProcessor
from langchain_openai import OpenAIEmbeddings
from claims.prompts import gpt_prompt_txt
from claims import logging
from claims.prompts import *
from claims.claim_generator import *
from claims.Tokenizer import *
from claims.PromptEngineering import *
from youtube_transcript_api._errors import *

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
email = "karthikamaravadi1234@gmail.com"
api_key = os.getenv('PUBMED_API_KEY')
st.set_page_config(page_title="CrediVerify", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #121212;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #121212;
    }
    .reportview-container {
        padding: 2rem;
    }
    .custom-title {
        background-color: #1e1e2e;
        color: #ffab00;
        font-size: 2.5rem;
        font-family: 'Roboto', sans-serif;
        font-weight: bold;
        padding: 1rem;
        text-align: center;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #1e88e5;
        color: white;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
    }
    .button-container {
        display: flex;
        align-items: center;
    }
    .button-container > * {
        margin-right: 10px;
    }
    input, textarea {
        background-color: #333;
        border: 2px solid #555;
        border-radius: 5px;
        padding: 10px;
        width: calc(100% - 130px);
        font-size: 1rem;
        color: #e0e0e0;
    }
    input:focus, textarea:focus {
        outline: none;
        border-color: #1e88e5;
    }
    .error-message, .success-message {
        border-radius: 5px;
        padding: 10px;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
    }
    .thumbnail {
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .footer {
        background-color: #1e1e2e;
        color: white;
        padding: 1rem;
        text-align: center;
        font-size: 0.9rem;
    }
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #333;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation functions
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page_name):
    st.session_state.page = page_name

def home_page():
    st.markdown('<div class="custom-title">WELCOME TO CREDIVERIFY</div>', unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([6, 1, 5])

    with col1:
        st.markdown("""
        🚀 **CrediVerify** is your go-to tool for fact-checking health-related YouTube videos! 🎥
        
        🧠 Using **Generative AI**, it extracts key claims from videos. Then, it taps into **PubMed** 📰 to fetch scientific articles, and with **Retrieval-Augmented Generation (RAG)** 🔍, it provides rock-solid claim validation.
        
        Say goodbye to misinformation and hello to trusted, data-backed insights! ✅
        """)
    
        # Layout for input and button
        with st.container():
            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            youtube_link = st.text_input("🎥 Enter YouTube Video Link:")
            os.environ['YOUTUBE_LINK'] = youtube_link
            search_button = st.button("🔍 Verify Link")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if search_button:
            video_id = extract_youtube_id(youtube_link)
            if not video_id:
                st.error("⚠️ Please provide a valid YouTube link for verification!")
            else:
                col3.image(f"http://img.youtube.com/vi/{video_id}/hqdefault.jpg", use_column_width=True, caption="Verify this thumbnail")
    
        if st.button("Get Detailed Claims and Validate"):
            st.session_state.page = "claims"

def Claims(ytlnk):
    # Clear previous content
    st.empty()

    col1, col2, col3 = st.columns([5,5,2])
    col3.button("🏠 Home", on_click=lambda: navigate_to("home"))

    col1.title("🔍 CLAIM VALIDATION", anchor="claim-validation")
    
    # Adding some space and styling
    st.markdown('<div style="font-size: 1.5rem; color: #ffab00; margin-bottom: 20px;">Claim Validation for Your Video</div>', unsafe_allow_html=True)
    
    placeholder = st.empty()
    
    with placeholder.container():
        try:
            video_id = extract_youtube_id(ytlnk)
            transcript_text = extract_transript_details(video_id)
            
            if transcript_text:
                summary = generate_gemini_content(transcript_text, YoutubeSummary_task)
            
            if summary:
                claims = generate_gemini_claims(summary, ClaimGenerator_task)
                
                if not health_video_check(Youtube_healh_check, claims):
                    st.error("⚠️ Only health-related videos in English with captions are allowed!")
                else:
                    lines = claims.strip().split("\n")
                    claims_list = [line.lstrip('* ').strip() for line in lines if line.startswith('* ')]
                    
                    st.markdown(f"### 📝 Found {len(claims_list)} claims in the video.", unsafe_allow_html=True)
                    
                    for i, claim in enumerate(claims_list, 1):
                        st.markdown(f"#### 🔹 **Claim {i}:**", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 1.2rem; color: #e0e0e0;'>{claim}</div>", unsafe_allow_html=True)
                        
                        # Claim validation process
                        response = generate_gemini_keywords(claim, Max_three_words_extraction)
                        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
                        topics = extract_keywords(response)
                        scraper = PubMedScraper(email, api_key)
                        date_range = '("2000/01/01"[Date - Create] : "2024/07/31"[Date - Create])'
                        df = scraper.run(topics, date_range)
                        
                        if df.empty:
                            result_qa = generate_chain_results1({"claim": claim})
                            st.markdown(f"#### ✅ **AI Validation Result for Claim {i}:**", unsafe_allow_html=True)
                            if isinstance(result_qa, dict):  # Check if result_qa is a dictionary
                                claims_formatted = {"claim": claims_list[i]}
                                result_qa = generate_chain_results1(claims_formatted)
                                logging.info(f"Final response: {result_qa}") 
                                st.write(result_qa)
                        else:
                            df_ranked = ranked_df(df, pd.read_csv('journal_rankings.csv'))
                            documents = load_documents(df_ranked)
                            in_memory_store = InMemoryVectorStore(documents, openai_embed_model)
                            custom_retriever = CustomRetriever(vectorstore=in_memory_store)
                            rag_processor = RAGQueryProcessor(custom_retriever=custom_retriever, gpt_prompt_txt=gpt_prompt_txt)
                            result_qa = rag_processor.process_query_retrieval_qa(claim)
                            st.markdown(f"#### 🔬 **PubMed Validation Result for Claim {i}:**", unsafe_allow_html=True)
                            st.write(result_qa)
        except AssertionError:
            st.error("⚠️ Invalid YouTube link!")




# Page routing logic
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'claims':
    Claims(os.environ.get("YOUTUBE_LINK"))
