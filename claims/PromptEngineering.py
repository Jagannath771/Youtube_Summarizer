from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv() ##Load all the new environment variables
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
# from google.generativeai.errors import GenerativeAIError

model_config1 = {
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
}


# Define your desired data structure - like a python data class.
scientific_validation_summary_task1="""Provide scientific Validation summary in less than 25 words:**
   - Conduct a thorough review of reputable medical research databases, such as PubMed, for studies related to the provided claim.
   - Prioritize peer-reviewed journals, with special emphasis on systematic reviews, cohort studies, meta-analyses and randomized controlled trials (RCTs) as they are high quality scientific evidence.
   - Do not consider case reports, case series, opinion pieces or observational studies and do not make up research papers as they are low quality evidence.
   - Evaluate the strength of evidence supporting the claim, as well as any contradictory or inconclusive findings."""

classification_task1= """Based on the retrieved high quality research papers from above task, classify the claim as one of the following:
**Scientific**: Supported by substantial, high-quality scientific evidence.
**Pseudo-science/Inconclusive**: Not supported by strong and credible evidence OR supported only by inconclusive scientific evidence, or contradicted by substantial evidence.
**Partially correct**: Supported by substantial scientific evidence but with significant caveats."""

research_summary_task1= """Research Summary in less than 25 words:Provide a concise summary of the research findings that support your classification."""
        
contradictory_claims_task1= """Contradictory Claims in less than 25 words: Identify if there are any scientifically supported evidence that contradicts the original claim or pose any health risks. 
If such evidence is found, explain why the contradicting claim is scientifically valid."""
 

class QueryResponse1(BaseModel):
    scientific_validation_summary: str = Field(description=scientific_validation_summary_task1)
    classification: str = Field(description=classification_task1)
    research_summary: str = Field(description=research_summary_task1)
    contradictory_claims: str = Field(description=contradictory_claims_task1)
    

# Set up a parser + inject instructions into the prompt template.
parser1 = JsonOutputParser(pydantic_object=QueryResponse1)

gpt_prompt_txt1= """
You are a medical researcher.Given the following health-related claim, generate the response based on the tasks specified in the following instructions:

claim= {claim}
Format Instructions: {format_instructions}
"""
gpt_prompt1 = PromptTemplate(
    template=gpt_prompt_txt1,
    input_variables=["claim"],
    partial_variables={"format_instructions": parser1.get_format_instructions()},
)
chatgpt1=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1, max_tokens= 500)
chain = (gpt_prompt1
           |
         chatgpt1
           |
         parser1)



def generate_chain_results1(claim_formatted):
    gpt_responses = chain.invoke(claim_formatted)
    response_dict = gpt_responses.dict() if hasattr(gpt_responses, 'dict') else gpt_responses
    row = {
        "Claim": claim_formatted,
        # "Scientific Validation Summary": response_dict.get("scientific_validation_summary", ""),
        "Classification": response_dict.get("classification", ""),
        "Research Summary": response_dict.get("research_summary", ""),
        "Contradictory Claims": response_dict.get("contradictory_claims", "")
    }
    return row