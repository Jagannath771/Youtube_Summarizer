import os
from dotenv import load_dotenv
load_dotenv() ##Load all the new environment variables
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
# from google.generativeai.errors import GenerativeAIError

model_config = {
  "temperature": 0.1,
  "top_p": 1,
  "top_k": 1,
}


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

def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro", generation_config=model_config)
    response = model.generate_content(prompt + transcript_text)
    return response.text

def generate_gemini_claims(summary, prompt):
    model = genai.GenerativeModel('gemini-pro', generation_config=model_config)
    model.temperature = 0
    response = model.generate_content(summary + prompt)
    return response.text


def generate_gemini_keywords(claims, keyword_prompt):
    try:
        # Initialize the generative model
        model = genai.GenerativeModel('gemini-pro', generation_config=model_config)
        
        # Generate the content based on the claims and keyword prompt
        response = model.generate_content(keyword_prompt + claims)
        
        # Check if the response has a valid 'text' part
        if hasattr(response, 'text'):
            return response.text
        else:
            raise ValueError("The response does not contain a valid 'text' attribute. Check the response object for details.")
    
    except ValueError as ve:
        # Handle the specific ValueError raised if the 'text' attribute is missing
        print(f"ValueError: {ve}")
        # You might want to log the response or take corrective action
        return None
    
    except GenerativeAIError as ge:
        # Handle any errors specific to the Generative AI library
        print(f"GenerativeAIError: {ge}")
        # Log the error or take corrective action
        return None

    except Exception as e:
        # Handle any other unforeseen errors
        print(f"An unexpected error occurred: {e}")
        # Log the error or take corrective action
        return None