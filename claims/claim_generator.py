import os
from dotenv import load_dotenv
load_dotenv() ##Load all the new environment variables
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI


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
    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

def generate_gemini_claims(summary, prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(summary+prompt)
    return response.text