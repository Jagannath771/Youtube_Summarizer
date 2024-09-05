import asyncio
import traceback

from claims.scraper import PubMedScraper
from claims.doc_loader import load_documents
from claims.utils import ranked_df
from claims.retrieval import InMemoryVectorStore, CustomRetriever
from claims.rag import RAGQueryProcessor
from langchain_openai import OpenAIEmbeddings

from claims import logging
from claims.prompts import *
from claims.claim_generator import *
from claims.Tokenizer import *
from claims.PromptEngineering import *
import google.generativeai as genai

load_dotenv()


async def scrape_pubmed(email, api_key, topics, date_range):
    async with PubMedScraper(email, api_key) as scraper:
        return await scraper.run(topics, date_range)


def process_claim(claim, email, api_key, api_key2, openai_embed_model):
    response = generate_gemini_keywords(claims=claim, keyword_prompt=Max_three_words_extraction)
    print(f"Processing claim: {claim}")

    if not response:
        print("No Valid Response received")
        return

    topics = extract_keywords(response)
    date_range = '("2010/03/01"[Date - Create] : "2024/07/31"[Date - Create])'

    logging.info("Scraping articles from Pubmed")
    df = asyncio.run(scrape_pubmed(email, api_key, topics, date_range))

    if len(df) == 0:
        print("LLM output")
        claims_formatted = {"claim": claim}
        result_qa = generate_chain_results1(claims_formatted)
    else:
        logging.info("Ranking articles based on journal rankings")
        df1 = pd.read_csv('Youtube_Summarizer//journal_rankings.csv')
        df_ranked = ranked_df(df, df1)

        logging.info("Loading documents into LangChain object")
        documents = load_documents(df_ranked)

        logging.info("Creating an in-memory vector store")
        in_memory_store = InMemoryVectorStore(documents, openai_embed_model)

        logging.info("Creating the custom retriever object")
        custom_retriever = CustomRetriever(vectorstore=in_memory_store)

        logging.info("Setting up the RAG chain")
        rag_processor = RAGQueryProcessor(custom_retriever=custom_retriever, gpt_prompt_txt=gpt_prompt_txt)

        logging.info("Running a query using RAG and GPT-4o-mini")
        result_qa = rag_processor.process_query_retrieval_qa(claim=claim)

    logging.info(f"Final response: {result_qa}")
    print(f"For the claim -> {claim}, the validation summary is")
    print(result_qa)


if __name__ == "__main__":
    try:
        genai.configure(api_key="AIzaSyD-W0BCGI-EwCAlbRYlCyHkAbV-e3PjeXo")
        transcript_text = extract_transript_details("https://www.youtube.com/watch?v=LDBeA9uJfI8")
        summary = generate_gemini_content(transcript_text, YoutubeSummary_task)
        claims = generate_gemini_claims(summary, ClaimGenerator_task)
        claims_list = [line.strip(' * , . 1234567890') for line in claims.split('\n') if line.strip()]

        email = "nithinpradeep38@gmail.com"
        api_key = "3c5ac885c60d2ae3a1c5b15a1ec162cfd409"
        api_key2 = "sk-qWdGe6C-13tm9IUhjQddDGl8r0fOBtmpdF1D-QB0aWT3BlbkFJrxC1-0kT0wrR_7zPJZ3vgcIYHpRcQvfOH3F5esqskA"

        os.environ['OPENAI_API_KEY'] = api_key2

        print(f"There are {len(claims_list)} claims in the given video. Below are the validations for each claim")

        openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

        for claim in claims_list:
            process_claim(claim, email, api_key, api_key2, openai_embed_model)
    except Exception:
        print(traceback.format_exc())
