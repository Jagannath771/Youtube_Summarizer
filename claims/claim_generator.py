import os
from dotenv import load_dotenv
load_dotenv() ##Load all the new environment variables
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi
import openai

model_config = {
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
}

def health_video_check(prompt, summary_text):
    
    model = genai.GenerativeModel("gemini-pro", generation_config=model_config)
    response = model.generate_content(prompt + summary_text)
    
    answer = response.text.strip().lower().split()
    answer=set(answer)
    print("Response:", answer)
    # Return True if the response contains "true", otherwise False
    return "true" in answer

def extract_transript_details(video_id):
    try:
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
        # print("Yes")
        transcript=""
        for i in transcript_text:
            transcript+=" "+i["text"]
        # print(transcript)

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
    except Exception as e:
        # Handle any other unforeseen errors
        print(f"An unexpected error occurred: {e}")
        # Log the error or take corrective action
        return None
    
def generate_gemini_results(claims, prompt):
    model = genai.GenerativeModel('gemini-pro', generation_config=model_config)
    response = model.generate_content(claims + prompt)
    return response.text
