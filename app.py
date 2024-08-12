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

# Load environment variables
load_dotenv()

# Configure Google Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Prompts
prompt = """You are a Youtube video summarizer. You will be taking the transcript text and summarizing the entire video to give a useful summary that provides an entire picture/idea to the user about the video. Please provide the summary of the text given here: """

prompt2 = """You are a Youtube Claims generator. You will be provided a summary of a Youtube video in detail, especially a health and fitness-related content of a Youtube video. You have to generate the top 5 claims for our health claims verification project. The claims need to be given in single-line points separated by * and in proper order with even medical terminology. Structure the sentences as Subject-verb-object (SVO) format. Please provide the claims for the text given here: """

# Function to extract transcript from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("v=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        st.error(f"Error retrieving transcript: {e}")
        return None

# Function to generate content using Gemini
def generate_gemini_content(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        return response.text
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

# Define the data structure for the response
class QueryResponse(BaseModel):
    scientific_validation_summary: str = Field(description="Provide scientific validation summary.")
    classification: str = Field(description="Classify the claim.")
    research_summary: str = Field(description="Provide research summary.")
    contradictory_claims: str = Field(description="Identify contradictory claims.")

# Set up a parser + inject instructions into the prompt template
parser = JsonOutputParser(pydantic_object=QueryResponse)

gpt_prompt_txt = """
You are a medical researcher. Given the following health-related claim, generate the response based on the tasks specified in the following instructions:

claim= {claim}
Format Instructions: {format_instructions}
"""

gpt_prompt = PromptTemplate(
    template=gpt_prompt_txt,
    input_variables=["claim"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Initialize the chat model
try:
    chatgpt = ChatOpenAI(
        model_name="gpt-4",  # Replace with the correct model name
        temperature=0.1, 
        max_tokens=500
    )
except Exception as e:
    st.error(f"Error initializing ChatOpenAI: {e}")
    chatgpt = None
chain = LLMChain(gpt_prompt | chatgpt | parser)

# Streamlit UI
st.title("Youtube Health Claims Validator")

youtube_link = st.text_input("Enter Youtube Video Link for a health-related video:")

if youtube_link:
    video_id = youtube_link.split("v=")[1]
    st.markdown("Verify the link with the thumbnail below and click the 'Get Detail Notes' Button")
    st.image(f"http://img.youtube.com/vi/{video_id}/hqdefault.jpg", use_column_width=True)

if st.button("Get Detail Claims and validate"):
    placeholder = st.empty()
    with placeholder.container():
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            summary = generate_gemini_content(transcript_text, prompt)
            if summary:
                claims = generate_gemini_content(summary, prompt2)
                if claims:
                    lines = claims.strip().split("\n")
                    claims_list = [line.lstrip('* ').strip() for line in lines if line.startswith('* ')]

                    claims_formatted = [{"claim": claim} for claim in claims_list]
                    if claims_formatted:
                        gpt_responses = chain.apply(claims_formatted)

                        # Prepare DataFrame for combined display
                        data = []
                        for claim, response in zip(claims_list, gpt_responses):
                            response_dict = response.dict() if hasattr(response, 'dict') else response
                            row = {
                                "Claim": claim,
                                "Classification": response_dict.get("classification", ""),
                                "Research Summary": response_dict.get("research_summary", ""),
                                "Contradictory Claims": response_dict.get("contradictory_claims", "")
                            }
                            data.append(row)

                        combined_df = pd.DataFrame(data)

                        # Apply custom styles for better readability
                        st.markdown("""
                                    <style>
                                    .dataframe td, .dataframe th {
                                        white-space: normal !important;
                                        word-wrap: break-word !important;
                                        max-width: 400px !important;
                                    }
                                    </style>
                                    """, unsafe_allow_html=True)

                        st.markdown("## Claims and Validation Responses")
                        st.dataframe(combined_df, use_container_width=True)
                    else:
                        st.write("No claims found.")
                else:
                    st.error("No claims generated from the summary.")
            else:
                st.error("Failed to generate summary.")
        else:
            st.error("Failed to retrieve transcript.")
