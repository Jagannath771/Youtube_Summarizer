from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import pandas as pd

"""
Not all articles fetched from pubmed maybe high quality. 
We want to ensure relevance and quality in the retrieval process.
Although, Pubmed is a reliable repository, it is better to trust articles from journals that are reliable publications.
We are mapping the SCImago Journal rank to the article
https://en.wikipedia.org/wiki/SCImago_Journal_Rank

Scimago Journal rank is a measure of the number of citations a journal receives in the top 10% of journals published in a given year.
"""

def get_best_match(row, choices, scorer):
    """
    The journal names in extracted articles may not exactly match the journal name in SCImago journal rankings. 
    So we use a fuzzy match to map them to extract the rank.
    """
    best_match, score = process.extractOne(row, choices, scorer=scorer)
    return pd.Series([best_match, score])

def ranked_df(df, df1):
    """
    Ranks the articles based on reputation scores of the journal. 
    df: The pubmed articles we fetched
    df1: The SCImago journal rank, a reputable measure of prestige of scholarly journals
    """
    choices= df1['Journal'].tolist() #create a list of names from rankings

    df[['best_match', 'score']]= df['Journal'].apply(lambda x: get_best_match(x, choices, scorer=fuzz.ratio)) #apply fuzzy match to journal names 

    merged_df = pd.merge(df, df1, left_on='best_match', right_on='Journal', how='left')
    merged_df.drop(columns= ['score', 'best_match', 'Journal_y','Journal_x', 'PMCID'], inplace=True)

    # Calculate the minimum and maximum values of the rank column
    min_rank = merged_df['Rank'].min()
    max_rank = merged_df['Rank'].max()

    # Apply min-max normalization,
    merged_df['normalized_rank'] = 1- (merged_df['Rank'] - min_rank) / (max_rank - min_rank)
    merged_df.drop(columns= ['Rank'], inplace= True)

    return merged_df
