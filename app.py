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
from streamlit_lottie import st_lottie
import requests

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

with st.sidebar:
    st.title("Navigation")
    selection = st.radio("Go to", ["Home", "About Us"])

def home_page():
    st.markdown("""<div class="custom-title">WELCOME TO CREDIVERIFY</div>
                <style>
        .stApp {
            background-image: url(https://images.pexels.com/photos/7470820/pexels-photo-7470820.jpeg);
            background-size: cover;
        }
        </style>
        """, unsafe_allow_html=True)
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
            search_button = st.button("🔍")
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
    st.markdown("""<div style="font-size: 1.5rem; color: #ffab00; margin-bottom: 20px;">Claim Validation for Your Video</div>
                <style>
        .stApp {
            background-image: url(https://images.pexels.com/photos/924824/pexels-photo-924824.jpeg);
            background-size: cover;
        }
        </style>
        """, unsafe_allow_html=True)
    
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
                    
                    for i, claim in enumerate(claims_list, 0):
                        st.markdown(f"#### 🔹 **Claim {i+1}:**", unsafe_allow_html=True)
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
                            st.markdown(f"#### ✅ **AI Validation Result for Claim {i+1}:**", unsafe_allow_html=True)
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
                            st.markdown(f"#### 🔬 **PubMed Validation Result for Claim {i+1}:**", unsafe_allow_html=True)
                            st.write(result_qa)
        except AssertionError:
            st.error("⚠️ Invalid YouTube link!")

def waitlist_page():
    st.empty()
    logo_url = r"Gemini_Generated_Image_ripzovripzovripz.jpeg"
    emoji_welcome = "🎉"
    emoji_waitlist = "📝"

    # Load Lottie animation
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_social = load_lottieurl("https://lottie.host/dd61585c-8bad-4ded-9184-fd344d8a8ed0/Maz6fBVacS.json")
    placeholder = st.empty()
    st.markdown("""
                <style>
        .stApp {
            background-image: url(https://images.pexels.com/photos/924824/pexels-photo-924824.jpeg);
            background-size: cover;
        }
        </style>
        """, unsafe_allow_html=True)
    with placeholder.container():
    # Create columns
        left_column, right_column = st.columns([2, 1])

        with left_column:
            st.title(f"We're Glad Your'e Here! {emoji_welcome}")
            st.image(logo_url, width=200)

            st.subheader("What is CrediVerify?")
            st.write(
                "CrediVerify is a cutting-edge tool designed to help you quickly assess the reliability of health advice from influencers on Social Media. Our app extracts content from videos and cross-references it with published peer-reviewed research to determine whether it's scientifically accurate or just pseudo-scientific junk. Stay informed and make smart decisions about the health content you consume!"
            )
            st.subheader("Our Mission")
            st.write(
                """At CrediVerify, our mission is to empower individuals with reliable information by bridging the gap between health-related content on social media and verified scientific research. In an era where misinformation can easily spread, we leverage cutting-edge Generative AI and advanced research techniques to analyze and validate the claims made in popular health videos. Our dedicated team is passionate about enhancing public knowledge and promoting well-being by providing accurate, data-backed insights. Through our innovative platform, we strive to create a more informed and healthier world, one verified claim at a time."""
                )

            st.subheader(f"Join our community! {emoji_waitlist}")
            st.write(
                "Be among the first to experience CrediVerify and revolutionize the way you consume health information. Sign up for our community to get exclusive updates and feature notes!"
            )

            # Embedding the contact form with HTML and CSS
            contact_form = """
            <form action="https://formsubmit.co/jagannathsai771@gmail.com" method="POST" style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <input type="hidden" name="_captcha" value="true">
                <label for="name" style="font-weight: bold; margin-top: 10px;">Name</label>
                <input type="text" id="name" name="name" placeholder="Your name" required style="width: 100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc;">
                <label for="email" style="font-weight: bold; margin-top: 10px;">Email</label>
                <input type="email" id="email" name="email" placeholder="Your email" required style="width: 100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc;">
                <label for="message" style="font-weight: bold; margin-top: 10px;">Message</label>
                <textarea id="message" name="message" placeholder="Your message" rows="5" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc;"></textarea>
                <button type="submit" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin-top: 10px;">Submit</button>
            </form>
            """
            st.markdown(contact_form, unsafe_allow_html=True)

        with right_column:
            st_lottie(lottie_social, height=400, key="social")


# Page routing logic

if selection=='About Us':
    st.session_state.page='about us'
if selection  == 'Home':
    if st.session_state.page=='claims':
        Claims(os.environ.get("YOUTUBE_LINK"))
    else:
        home_page()
# if st.session_state.page == 'claims':
#     Claims(os.environ.get("YOUTUBE_LINK"))
if selection  == "About Us":
    waitlist_page()
    