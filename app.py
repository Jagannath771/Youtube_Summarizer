import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv() ##Load all the new environment variables
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
prompt="""You are a Youtube video summarizer. You will be taking the trascript text and summarizing the entire video and give a useful summary which gives an entire picture/idea to the user
about the video. Please provide the summary of the text given here : """

##Getting the transcript data from yt videos
def extract_transript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        # print(video_id)
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
        transcript=""
        for i in transcript_text:
            transcript+=" "+i["text"]

    except Exception as e:
        raise e
    return transcript

#Getting the summary based on prompt from Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

prompt2='''You are a Youtube Claims generator. You will be provided a summary of a youtube video in detail, especially a health and fitness related content of a youtube video. You have to 
generate top 5 claims for our health claims verification project. The claims needs to be given in single line points separated by * and in proper order with even medical terminology. Structure the sentences as Subject-verb-object (SVO) format. Please provide the claims for the text
given here :'''

def generate_gemini_claims(summary, prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(prompt+summary)
    return response.text

# youtube_link=str(input("Give Link: "))
# transcript_text=extract_transript_details(youtube_link)

# print(generate_gemini_content(transcript_text, prompt))

# Define your desired data structure - like a python data class.
scientific_validation_summary_task="""Provide scientific Validation summary in less than 25 words:**
   - Conduct a thorough review of reputable medical research databases, such as PubMed, for studies related to the provided claim.
   - Prioritize peer-reviewed journals, with special emphasis on systematic reviews, cohort studies, meta-analyses and randomized controlled trials (RCTs) as they are high quality scientific evidence.
   - Do not consider case reports, case series, opinion pieces or observational studies and do not make up research papers as they are low quality evidence.
   - Evaluate the strength of evidence supporting the claim, as well as any contradictory or inconclusive findings."""

classification_task= """Based on the retrieved high quality research papers from above task, classify the claim as one of the following:
**Scientific**: Supported by substantial, high-quality scientific evidence.
**Pseudo-science/Inconclusive**: Not supported by strong and credible evidence OR supported only by inconclusive scientific evidence, or contradicted by substantial evidence.
**Partially correct**: Supported by substantial scientific evidence but with significant caveats."""

research_summary_task= """Research Summary in less than 25 words:Provide a concise summary of the research findings that support your classification."""
        
contradictory_claims_task= """Contradictory Claims in less than 25 words: Identify if there are any scientifically supported evidence that contradicts the original claim or pose any health risks. 
If such evidence is found, explain why the contradicting claim is scientifically valid."""
 

class QueryResponse(BaseModel):
    scientific_validation_summary: str = Field(description=scientific_validation_summary_task)
    classification: str = Field(description=classification_task)
    research_summary: str = Field(description=research_summary_task)
    contradictory_claims: str = Field(description=contradictory_claims_task)
    

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=QueryResponse)

gpt_prompt_txt= """
You are a medical researcher.Given the following health-related claim, generate the response based on the tasks specified in the following instructions:

claim= {claim}
Format Instructions: {format_instructions}
"""
gpt_prompt = PromptTemplate(
    template=gpt_prompt_txt,
    input_variables=["claim"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chatgpt=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, max_tokens= 500)
chain = (gpt_prompt
           |
         chatgpt
           |
         parser)


st.title("Youtube Health Claims Validator")
youtube_link=st.text_input("Enter Youtube Video Link for a health related video:")

if youtube_link:
    video_id=youtube_link.split("=")[1]
    # print(video_id)
    st.markdown("Verify the link with the thumbnail below and click the Get Detail Notes Button")
    st.image(f"http://img.youtube.com/vi/{video_id}/hqdefault.jpg", use_column_width=True)
if st.button("Get Detail Claims and validate"):
    placeholder= st.empty()
    with placeholder.container():
        transcript_text=extract_transript_details(youtube_link)

        if transcript_text: 
            summary=generate_gemini_content(transcript_text, prompt)
            claims=generate_gemini_claims(summary, prompt2)
            if claims:
                lines= claims.strip().split("\n")
                claims_list= [line.lstrip('* ').strip() for line in lines if line.startswith('* ')]
                claims_formatted= [{"claim": claim} for claim in claims_list]
    
                if claims_formatted:
                    gpt_responses = chain.map().invoke(claims_formatted)
                    
                    # Prepare DataFrame for combined display
                    data = []
                    for claim, response in zip(claims_list, gpt_responses):
                        response_dict = response.dict() if hasattr(response, 'dict') else response
                        row = {
                            "Claim": claim,
                           # "Scientific Validation Summary": response_dict.get("scientific_validation_summary", ""),
                            "Classification": response_dict.get("classification", ""),
                            "Research Summary": response_dict.get("research_summary", ""),
                            "Contradictory Claims": response_dict.get("contradictory_claims", "")
                        }
                        data.append(row)
                    
                    combined_df = pd.DataFrame(data)

                    st.markdown("""
                                <style>
                                .dataframe td, .dataframe th {
                                    white-space: normal !important;
                                    word-wrap: break-word !important;
                                    max-width: 400px !important;  /* Adjust this value as needed */
                                }
                                </style>
                                """, unsafe_allow_html=True)
                    
                    st.markdown("## Claims and Validation Responses")
                    st.dataframe(combined_df, use_container_width=True)
                else:
                    st.write("No claims found.")

            else:
                st.error("No Claims generated from summary")
        else:
            st.error("Failed to retrieve transcript.")
    

    
