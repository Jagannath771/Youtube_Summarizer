import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text):
    """
    Extracts keywords from a text containing bullet points.

    Parameters:
        text (str): The input text containing bullet points.

    Returns:
        List[str]: A list of keywords extracted from the text.
    """
    try:
        # Split the text into lines
        lines = text.strip().split('\n')
        
        # Process each line to remove bullet points and extra whitespace
        keywords = [line.strip('-•* \t') for line in lines if line.strip()]
    
        return keywords
    except AttributeError as AE:
        print(f"Attribute Error:{AE}")
        return []


import re

def extract_youtube_id(url):
    # List of regular expressions to match various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and short URL
        r'(?:embed\/|v\/|youtu.be\/)([0-9A-Za-z_-]{11})',  # Embedded and youtu.be URL
        r'(?:watch\?)?(?:feature=player_embedded&)?(?:v=)?(?:video_ids=)?([0-9A-Za-z_-]{11})',  # Various watch URLs
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',  # YouTube Shorts
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

