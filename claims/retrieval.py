from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from claims import logging

"""
We download documents and create vector on-the-fly which will used for response generation.
Since the documents vary from video to video, no incentive to store them in a vector DB. 
So using an in memory vector store for response generation.

We use the vector store to rank documents based on cosine similarity, followed by a ranking based on journal quality. 
This is as measured by the normalized rank.
"""

class InMemoryVectorStore:
    """
    The standard vector store that uses a cosine similarity measure for ranking documents based on similarity with query
    """
    def __init__(self, documents, embedding_function):
        self.documents = documents
        self.embedding_function = embedding_function
        self.embeddings = None

    def create_embeddings(self):
        try:
            texts = [doc.page_content for doc in self.documents]
            self.embeddings = self.embedding_function.embed_documents(texts)
        except Exception as e:
            logging.info(f"Error creating embeddings: {e}")
            return None

    def similarity_search_with_score(self, query, k=5):
        """
        Get the top k documents with the highest cosine similarity to the query,
        
        :param query: The input query string
        """
        try:
            query_embedding = self.embedding_function.embed_query(query)
            
            if self.embeddings is None:
                self.create_embeddings()
            
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            sorted_indices = np.argsort(similarities)[::-1][:k]
            
            return [(self.documents[i], similarities[i]) for i in sorted_indices]
        except Exception as e:
            logging.info(f"Error during similarity search: {e}")
            return []

    def _select_relevance_score_fn(self):
        return lambda score: score  # return score as is

class CustomRetriever(VectorStoreRetriever):
    """
    Custom retriever that uses normalized rank of journal to re-rank the retrieved similar documents.
    """
    vectorstore: InMemoryVectorStore
    topk: int = 5
    similarity_weight: float = 0.8
    search_type: str = "similarity"
    search_kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get the top k documents reranked based on the journal ranking
        
        :param query: The input query string
        """
        try:
            # Get similarity conversion function
            cosine_sim = self.vectorstore._select_relevance_score_fn()

            # Get top k documents with lowest cosine distance
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.topk)
            docs, scores = zip(*docs_and_scores)
            scores = [cosine_sim(score) for score in scores]
            min_score = min(scores)
            max_score = max(scores)
            normalized_scores = [(score-min_score)/(max_score-min_score) for score in scores]
            
            # Calculate final score considering both similarity and rank
            for doc, score, normalized_score in zip(docs, scores, normalized_scores):
                # Normalize the cosine similarity scores
                doc.metadata['normalized_score'] = normalized_score
                rank = doc.metadata['normalized_rank']
                final_score = self.similarity_weight * normalized_score + (1 - self.similarity_weight) * rank
                doc.metadata['final_score'] = final_score

            # Sort documents by final score
            sorted_docs = sorted(docs, key=lambda x: x.metadata['final_score'], reverse=True)

            return sorted_docs[:self.topk]
        except Exception as e:
            logging.info(f"Error during reranking based on journal rank: {e}")
            return []