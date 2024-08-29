import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text):
  """
  Extracts keywords from a given text.

  Args:
    text: The input text.

  Returns:
    A list of keywords.
  """

  # Tokenization
  tokens = word_tokenize(text)

  # Remove stop words
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word not in stop_words]

  # Apply stemming or lemmatization (optional)
  # stemmer = PorterStemmer()
  # filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]

  return filtered_tokens

def extract_keywords_and_tfidf(claims):
  """
  Extracts keywords from a list of claims and calculates TF-IDF vectors.

  Args:
    claims: A list of claim strings.

  Returns:
    A tuple containing a list of keywords and a TF-IDF matrix.
  """

  # Assuming you have a function to extract keywords (e.g., using NLTK)
  keywords_list = [extract_keywords(claim) for claim in claims]

  # Join keywords into a single string for each claim
  claim_texts = [" ".join(keywords) for keywords in keywords_list]

  # Create a TF-IDF vectorizer
  vectorizer = TfidfVectorizer()

  return keywords_list

