import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv() ##Load all the new environment variables
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi

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
generate claims for our health claims verification project, the claims needs to given in points and in proper order with even medical terminology. Please provide the claims for the text
given here :'''

def generate_gemini_claims(summary, prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(prompt+summary)
    return response.text

# youtube_link=str(input("Give Link: "))
# transcript_text=extract_transript_details(youtube_link)

# print(generate_gemini_content(transcript_text, prompt))



st.title("Youtube Health Claims Generator")
youtube_link=st.text_input("Enter Youtube Video Link for a health related video:")

if youtube_link:
    video_id=youtube_link.split("=")[1]
    # print(video_id)
    st.markdown("Verify the link with the thumbnail below and click the Get Detail Notes Button")
    st.image(f"http://img.youtube.com/vi/{video_id}/hqdefault.jpg", use_column_width=True)
if st.button("Get Detail Claims"):
    transcript_text=extract_transript_details(youtube_link)

    if transcript_text: 
        summary=generate_gemini_content(transcript_text, prompt)
        claims=generate_gemini_claims(summary, prompt2)
        # print(video_id)
        st.markdown("## Claims stated in the Video")
        st.write(claims)
