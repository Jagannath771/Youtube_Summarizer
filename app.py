import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import langchain_openai
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
import os
import pandas as pd
from claims.prompts import gpt_prompt_txt
from dotenv import load_dotenv
from claims import  logging
from claims.prompts import *
from claims.claim_generator import *
from claims.Tokenizer import *
from claims.PromptEngineering import *
from youtube_transcript_api._errors import *

# Load environment variables
load_dotenv()
# langchain_openai.configure(api_key=os.getenv("OPENAI_API_KEY"))
email = "karthikamaravadi1234@gmail.com"
api_key = os.getenv('PUBMED_API_KEY')
st.set_page_config(layout="wide")
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page_name):
    st.session_state.page = page_name


def home_page():
    os.environ["YOUTUBE_LINK"]=''
    st.markdown("""
    <style>
    .custom-title {
        background-color: White; /* Green background */
        color: limegreen;
        font-size: 30px;
        font-family: cursive;
        font-weight:bold;
        padding: 30px;
        text-align: center;
    }
    </style>
    <div class="custom-title">WELCOME TO CREDIVERIFY</div>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([6, 1, 5])  # Adjust the proportions here
    youtube_link = col1.text_input("Enter Youtube Video Link for a health-related video:")
    os.environ['YOUTUBE_LINK'] = youtube_link
    col2.write('')
    col2.write('')
    search_button=col2.button("🔍")
    try:
        if search_button:
            video_id = extract_youtube_id(youtube_link)
            col3.markdown("Verify the link with the thumbnail below and click the 'Get Detail Notes' Button")
            col3.image(f"http://img.youtube.com/vi/{video_id}/hqdefault.jpg", use_column_width=True)
    except IndexError:
        st.error("Please provide a youtube link for validation!")
    if col1.button("Get Detail Claims and validate",type="primary"):
        navigate_to("claims")

def Claims(ytlnk):
    col1, col2, col3 = st.columns([5,5,2])
    col3.button("Go to Home Page", on_click=navigate_to("home"), type="primary")
    col1.title("CLAIMS")
    placeholder= st.empty()
    with placeholder.container():
        try:
            video_id = extract_youtube_id(ytlnk)
            transcript_text = extract_transript_details(video_id)
            if transcript_text:
                summary = generate_gemini_content(transcript_text, YoutubeSummary_task)
            if summary:
                claims = generate_gemini_claims(summary, ClaimGenerator_task)
                if not health_video_check(Youtube_healh_check,claims):
                    st.error("Please provide link of only a health related video in English!")
                    st.stop()
                if claims:
                    lines = claims.strip().split("\n")
                    claims_list = [line.lstrip('* ').strip() for line in lines if line.startswith('* ')]

                    # claims_formatted = [{"claim": claim} for claim in claims_list]
                    # st.write(claims_list)
                    st.write(f"There are {len(claims_list)} claims in the given video. Below are the validation for each claim")
                    for i in range(len(claims_list)): 
                        st.write(f"-> Claim {i+1}")
                        response= generate_gemini_keywords(claims= claims_list[i], keyword_prompt=Max_three_words_extraction)
                        print(i+1)
                        # if response:
                        #     print(response)
                        # else:
                        #     print("No Valid Response recieved")
                        #     continue
                    # OpenAI embedding
                        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
                        # claim="Coffee helps reduce chances of liver cancer "
                        logging.info(f"Claim being assessed: {claims_list[i]}")
                        topics = extract_keywords(response)
                        date_range = '("2010/03/01"[Date - Create] : "2024/07/31"[Date - Create])'
                        
                        logging.info("Scraping articles from Pubmed")
                        # print(2)
                        st.write("Extracting Journals from PubMed ....")
                        scraper = PubMedScraper(email, api_key)
                        # print(3)
                        df = scraper.run(topics, date_range)
                        if len(df)==0:
                            st.write("AI Validated Output")
                            claims_formatted= {"claim": claims_list[i]}
                            result_qa=generate_chain_results1(claims_formatted)
                            logging.info(f"Final response: {result_qa}") 
                            st.write(f"For the claim ->{claims_list[i]}, the validation summary is")
                            st.write(result_qa)
                        else:
                            # print(df)
                            # print(df.head())
                            logging.info("Ranking articles based on journal rankings")
                            df1= pd.read_csv('journal_rankings.csv')
                            # print(df1)
                            df_ranked= ranked_df(df,df1)
                            # print(df_ranked)
                            logging.info("Loading documents into LangChain object")
                            documents= load_documents(df_ranked)
                            
                            logging.info("Creating an in-memory vector store")  # create an in-memory vector store with the documents and embeddings
                            in_memory_store = InMemoryVectorStore(documents, openai_embed_model)
                            
                            logging.info("Creating the custom retriever object")
                            custom_retriever= CustomRetriever(vectorstore= in_memory_store)
                            
                            logging.info("Setting up the RAG chain")  # setup the RAG chain using the custom retriever and GPT-4o-mini prompt
                            rag_processor= RAGQueryProcessor(custom_retriever=custom_retriever,gpt_prompt_txt= gpt_prompt_txt)
                            
                            logging.info("Running a query using RAG and GPT-4o-mini")  # run a query using RAG and GPT-4o-mini to validate the claims.
                            result_qa= rag_processor.process_query_retrieval_qa(claim=claims_list[i])
                            logging.info(f"Final response: {result_qa}")  # print the final response from the RAG chain and GPT-4o-mini prompt.  # print the final response from the RAG chain and GPT-4o-mini prompt.  # print the final response from the RAG chain and GPT-4o-mini prompt.  # print the final response from the RAG chain and GPT-4o-mini prompt.  # print the final response from the R
                            st.write("PubMed RAG Output")
                            st.write(f"For the claim ->{claims_list[i]}, the validation summary is")
                            st.write(result_qa)
        except IndexError:
            st.error("Please provide a youtube link for validation!")
        except TranscriptsDisabled:
            st.error("Trascripts are disabled for this video. Cant generate claims as of now.")
        except NoTranscriptFound:
            st.error("No Trascript Found. Please provide a valid video with English audio!")
        

if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'claims':
    Claims(os.environ.get("YOUTUBE_LINK"))