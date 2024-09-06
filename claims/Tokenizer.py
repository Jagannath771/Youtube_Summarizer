import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# def remove_punctuation(text):
#   """
#   Removes punctuation characters from text.

#   Args:
#       text: The input text.

#   Returns:
#       The text with punctuation removed.
#   """
#   punctuation = set([',', '.', '-'])
#   return ''.join(char for char in text if char not in punctuation)

# def extract_keywords(text):
#   """
#   Extracts keywords from a given text.

#   Args:
#       text: The input text.

#   Returns:
#       A list of keywords.
#   """

#   # Preprocess text by removing punctuation
#   text = remove_punctuation(text)

#   # Tokenization
#   tokens = word_tokenize(text)

#   # Remove stop words
#   stop_words = set(stopwords.words('english'))
#   filtered_tokens = [word.lower() for word in tokens if word not in stop_words]

#   # Apply stemming or lemmatization (optional)
#   # stemmer = PorterStemmer()
#   # filtered_tokens = [stemmer.stem(word) for word in filtered_tokens]

#   return filtered_tokens

# def extract_keywords_and_tfidf(claims):
#   """
#   Extracts keywords from a list of claims and calculates TF-IDF vectors.

#   Args:
#       claims: A list of claim strings.

#   Returns:
#       A tuple containing a list of keywords and a TF-IDF matrix.
#   """

#   # Preprocess claims by removing punctuation
#   claims_preprocessed = [remove_punctuation(claim) for claim in claims]

#   # Extract keywords using the updated extract_keywords function
#   keywords_list = [extract_keywords(claim) for claim in claims_preprocessed]

#   # Join keywords into a single string for each claim
#   claim_texts = [" ".join(keywords) for keywords in keywords_list]

#   # Create a TF-IDF vectorizer
#   vectorizer = TfidfVectorizer()

#   return keywords_list

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

def is_health_video(claims):
    health_keywords = [
    # General Health
    "health", "wellness", "medicine", "medical", "healthcare", "hospital", "clinic", "doctor",
    "physician", "surgeon", "therapy", "treatment", "diagnosis", "prescription", "symptoms",
    "surgery", "recovery", "rehabilitation", "prevention", "cure", "health risk", "public health",
    "mental health", "physical health", "chronic disease", "infection", "virus", "bacteria",
    "immunity", "immune system",

    # Fitness and Exercise
    "fitness", "exercise", "workout", "gym", "physical activity", "training", "strength training",
    "cardio", "aerobics", "yoga", "pilates", "flexibility", "endurance", "hiit", "bodybuilding",
    "crossfit", "sports", "athletic", "running", "cycling", "swimming", "weight loss", "fat burning",
    "muscle gain",

    # Nutrition and Diet
    "nutrition", "diet", "healthy eating", "balanced diet", "calories", "macronutrients", 
    "micronutrients", "vitamins", "minerals", "protein", "carbohydrates", "fats", "fiber",
    "antioxidants", "superfoods", "organic food", "vegan", "vegetarian", "gluten-free", 
    "keto", "paleo", "intermittent fasting", "supplement", "vitamin d", "omega-3", "hydration",
    "probiotics", "detox",

    # Mental Health
    "mental health", "psychology", "psychiatry", "counseling", "stress", "anxiety", "depression",
    "bipolar disorder", "schizophrenia", "cognitive behavioral therapy", "cbt", "mindfulness",
    "meditation", "relaxation", "self-care", "well-being", "mood", "emotional health", "ptsd",
    "post-traumatic stress disorder", "ocd", "obsessive-compulsive disorder", "adhd",
    "attention deficit hyperactivity disorder", "resilience", "coping strategies", "trauma",

    # Specific Health Conditions
    "diabetes", "hypertension", "heart disease", "cancer", "stroke", "asthma", "arthritis",
    "alzheimer’s disease", "dementia", "obesity", "hiv", "aids", "tuberculosis", "cholesterol",
    "blood pressure", "cardiovascular", "respiratory", "gastrointestinal", "digestive health",
    "liver disease", "kidney disease", "skin conditions", "allergies", "autoimmune disorders",
    "chronic pain", "migraine", "sleep disorders", "insomnia",

    # Reproductive and Sexual Health
    "reproductive health", "sexual health", "pregnancy", "maternity", "prenatal", "postnatal",
    "fertility", "contraception", "birth control", "menopause", "menstruation", "hormones",
    "sexuality", "std", "sexually transmitted disease", "hiv", "hpv", "human papillomavirus",
    "sexual wellness",

    # Child and Adolescent Health
    "pediatrics", "child health", "infant care", "vaccination", "immunization", "child nutrition",
    "growth and development", "parenting", "adolescent health", "school health", "teen wellness",
    "child psychology", "learning disabilities",

    # Geriatric Health
    "geriatrics", "elderly care", "aging", "senior health", "dementia", "alzheimer’s",
    "osteoporosis", "mobility", "longevity", "palliative care", "end-of-life care",

    # Public Health and Safety
    "public health", "epidemic", "pandemic", "quarantine", "vaccination", "immunization",
    "hygiene", "sanitation", "health policy", "healthcare reform", "environmental health",
    "occupational health", "health insurance", "healthcare access", "health disparities",
    "global health", "health education", "health literacy"
]
    herbal_medications_and_plants = [
    "Aloe Vera",
    "Ginger",
    "Garlic",
    "Echinacea",
    "Turmeric",
    "Ginseng",
    "Chamomile",
    "Peppermint",
    "Lavender",
    "Milk Thistle",
    "St. John's Wort",
    "Valerian Root",
    "Saw Palmetto",
    "Feverfew",
    "Cranberry",
    "Elderberry",
    "Ashwagandha",
    "Holy Basil (Tulsi)",
    "Licorice Root",
    "Dandelion",
    "Calendula",
    "Hibiscus",
    "Yarrow",
    "Catnip",
    "Goldenseal",
    "Slippery Elm",
    "Red Clover",
    "Bilberry",
    "Ginkgo Biloba",
    "Fenugreek",
    "Dong Quai",
    "Moringa",
    "Neem",
    "Sage",
    "Thyme",
    "Rosemary",
    "Burdock Root",
    "Nettle",
    "Basil",
    "Lemon Balm",
    "Cinnamon",
    "Cardamom",
    "Clove",
    "Fennel",
    "Hawthorn",
    "Kava Kava",
    "Lemon Grass",
    "Passionflower",
    "Rhodiola",
    "Saffron",
    "Schisandra",
    "Spirulina",
    "Yohimbe",
    "Yerba Mate",
    "Devil’s Claw",
    "Red Raspberry Leaf",
    "Oregano",
    "Marshmallow Root",
    "Horse Chestnut",
    "Evening Primrose",
    "Black Cohosh",
    "Blue Cohosh",
    "Arnica",
    "Comfrey",
    "Witch Hazel",
    "Licorice",
    "Cayenne",
    "Brahmi",
    "Tulsi",
    "Shatavari"
]
    health_keywords.extend(herbal_medications_and_plants)
    sentence_words = claims.lower().split()
    count = 0
    health_keywords=set(health_keywords)
    
    # Iterate through the word list and count matches in the sentence
    for word in sentence_words:
        if word.lower() in health_keywords:
            print(f"The medical terminology found in the video {word}")
            count += 1
            if count==3:
                return True
    return False


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