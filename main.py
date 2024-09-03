from claims.scraper import PubMedScraper
from claims.doc_loader import load_documents
from claims.utils import ranked_df
from claims.retrieval import InMemoryVectorStore, CustomRetriever
from claims.rag import RAGQueryProcessor
from langchain_openai import OpenAIEmbeddings
import os
import pandas as pd
from claims.prompts import gpt_prompt_txt
from dotenv import load_dotenv
from claims import  logging
from claims.prompts import *
from claims.claim_generator import *
from claims.Tokenizer import *
from claims.PromptEngineering import *
load_dotenv()

if __name__ == "__main__":
    genai.configure(api_key="AIzaSyD-W0BCGI-EwCAlbRYlCyHkAbV-e3PjeXo")
    transcript_text = extract_transript_details("https://www.youtube.com/watch?v=LDBeA9uJfI8")
    summary=generate_gemini_content(transcript_text,YoutubeSummary_task)
    claims=generate_gemini_claims(summary, ClaimGenerator_taks)
    # print(claims)
    claims_list = [line.strip(' * , . 1234567890') for line in claims.split('\n') if line.strip()]
    
# Print the resulting array
    # print(claims_list)
    # keywords_list = extract_keywords_and_tfidf(claims_list)
    # print(keywords_list)
    email = "nithinpradeep38@gmail.com"
    api_key = "3c5ac885c60d2ae3a1c5b15a1ec162cfd409"#os.getenv('PUBMED_API_KEY')
    # print(api_key)
    # Fetch the API key from environment variables
    api_key2 = "sk-qWdGe6C-13tm9IUhjQddDGl8r0fOBtmpdF1D-QB0aWT3BlbkFJrxC1-0kT0wrR_7zPJZ3vgcIYHpRcQvfOH3F5esqskA"#os.getenv('OPENAI_API_KEY')

    # if api_key2 is None:
    #     raise ValueError("The environment variable 'OPENAI_API_KEY' is not set. Please set it before running the script.")

    # Set the API key for your application
    os.environ['OPENAI_API_KEY'] = api_key2
    # os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
    print(f"There are {len(claims_list)} claims in the given video. Below are the validation for each claim")
    for i in range(len(claims_list)): 
        response= generate_gemini_keywords(claims= claims_list[i], keyword_prompt=Max_three_words_extraction)
        print(i+1)
        # if response:
        #     print(response)
        # else:
        #     print("No Valid Response recieved")
        #     continue
    # OpenAI embedding
        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')
        # claim="Coffee helps reduce chances of liver cancer "
        logging.info(f"Claim being assessed: {claims_list[i]}")
        topics = extract_keywords(response)
        date_range = '("2010/03/01"[Date - Create] : "2024/07/31"[Date - Create])'
        
        logging.info("Scraping articles from Pubmed")
        # print(2)
        scraper = PubMedScraper(email, api_key)
        # print(3)
        df = scraper.run(topics, date_range)
        if len(df)==0:
            print("LLM output")
            claims_formatted= {"claim": claims_list[i]}
            result_qa=generate_chain_results1(claims_formatted)
            logging.info(f"Final response: {result_qa}") 
            print(f"For the claim ->{claims_list[i]}, the validation summary is")
            print(result_qa)
        else:
            # print(df)
            # print(df.head())
            logging.info("Ranking articles based on journal rankings")
            df1= pd.read_csv('Youtube_Summarizer//journal_rankings.csv')
            # print(df1)
            df_ranked= ranked_df(df,df1)
            # print(df_ranked)
            logging.info("Loading documents into LangChain object")
            documents= load_documents(df_ranked)
            
            logging.info("Creating an in-memory vector store")  # create an in-memory vector store with the documents and embeddings
            in_memory_store = InMemoryVectorStore(documents, openai_embed_model)
            
            logging.info("Creating the custom retriever object")
            custom_retriever= CustomRetriever(vectorstore= in_memory_store)
            
            logging.info("Setting up the RAG chain")  # setup the RAG chain using the custom retriever and GPT-4o-mini prompt
            rag_processor= RAGQueryProcessor(custom_retriever=custom_retriever,gpt_prompt_txt= gpt_prompt_txt)
            
            logging.info("Running a query using RAG and GPT-4o-mini")  # run a query using RAG and GPT-4o-mini to validate the claims.
            result_qa= rag_processor.process_query_retrieval_qa(claim=claims_list[i])
            logging.info(f"Final response: {result_qa}")  # print the final response from the RAG chain and GPT-4o-mini prompt.  # print the final response from the RAG chain and GPT-4o-mini prompt.  # print the final response from the RAG chain and GPT-4o-mini prompt.  # print the final response from the RAG chain and GPT-4o-mini prompt.  # print the final response from the R
            print(f"For the claim ->{claims_list[i]}, the validation summary is")
            print(result_qa)
    
    