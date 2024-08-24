from langchain_core.documents import Document
from claims import logging

def load_documents(df):
    """
    Converts a Dataframe of research papers into a list of LangChain document objects.

    Parameters:
    df: pandas.DataFrame of title, abstract, Conclusions, PMID, URL and normalized rank.
    Returns: A list of LangChain document objects with content being title, abstract and conclusions.
             Metadata: PMID, URL and normalized rank of journal
    """
    try:
        documents= []
        for index, row in df.iterrows():
            content = f"Title: {row['Title']}\n\nAbstract: {row['Abstract']}\n\nConclusions: {row['Conclusions']}"
            doc = Document(
            page_content=content,
            metadata={
                "PMID": row['PMID'],
                "URL": row['URL'],
                "normalized_rank": row['normalized_rank']  # Include the rank in metadata
            }
        )
            documents.append(doc)

        return documents
    except Exception as e:
        logging.info(f"Error loading documents: {e}")
