# scraper.py
from Bio import Entrez
import xml.etree.ElementTree as ET
import aiohttp
from aiohttp import ClientSession
import asyncio
import pandas as pd
import time
from claims import logging

"""
A scrapped to extract pubmed articles. 
We will use an aiohttp client/server framework for asynchronous processing of requests

We take list of claim keywords as input and do the following.
1. Build PubMed query for given topics and date range.
2. Fetch articles using Entrez.esearch. Unless free, we can only extract the abstract of the papers.
3. Fetch PMCID, if exists. These are the full articles available for free on pubmed. 
4. Extract conclusions if PMCID exists.
5. Store the abstract, title, journal, conclusion (if exists) into a pandas dataframe.
"""

class PubMedScraper:
    def __init__(self, email, api_key):
        self.email = email
        self.api_key = api_key
        Entrez.email = email

    def build_query(self, topics, date_range):
        queries = []
        if topics:
            topic_queries = ['{}[Title/Abstract]'.format(topic) for topic in topics]
            queries.append('(' + ' AND '.join(topic_queries) + ')')
        full_query = ' AND '.join(queries) + ' AND ' + date_range
        return full_query

    async def fetch_pmc_conclusions(self, pmcid, session):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        efetch_url = f"{base_url}efetch.fcgi?db=pmc&id={pmcid}&retmode=xml&api_key={self.api_key}"

        try:
            async with session.get(efetch_url) as response:
                if response.status != 200:
                    print(f"Error fetching PMC article: HTTP {response.status}")
                    return f"Failed to fetch PMC content for {pmcid}: HTTP {response.status}"
                content = await response.text()
                #print(f"Received XML content for {pmcid}: {content[:200]}...")

                root = ET.fromstring(content)

                conclusions = ""
                for section in root.findall(".//sec"):
                    section_title = section.find("title")
                    if section_title is not None and section_title.text and "conclusion" in section_title.text.lower():
                        for p in section.findall(".//p"):
                            conclusions += ET.tostring(p, encoding='unicode', method='text') + "\n"
                        break

                if not conclusions:
                    logging.info(f"Conclusions section not found for {pmcid}")
                    return f"Conclusions section not found in the article for {pmcid}."

                return conclusions.strip()
     
        except Exception as e:
            logging.info(f"Error fetching PMC article {pmcid}: {e}")
            return f"Failed to fetch PMC content for {pmcid}: {str(e)}"

    async def fetch_pmcid_and_conclusions(self, pmid, session):
        try:
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            elink_url = f"{base_url}elink.fcgi?dbfrom=pubmed&db=pmc&id={pmid}&retmode=xml&api_key={self.api_key}"
            
            async with session.get(elink_url) as response:
                if response.status != 200:
                    return None, "No PMC article available"
                
                content = await response.text()
                root = ET.fromstring(content)
                pmcid_element = root.find(".//LinkSetDb/Link/Id")
                if pmcid_element is not None:
                    pmcid = f"PMC{pmcid_element.text}"
                    conclusions = await self.fetch_pmc_conclusions(pmcid, session)
                    return pmcid, conclusions
                else:
                    return None, "No PMC article available"
        except Exception as e:
            logging.info(f"Error fetching PMC article: {e}")
            return None, "Failed to fetch PMC article"

    async def fetch_pubmed_record(self, pmid, session):
        handle = Entrez.efetch(db='pubmed', id=pmid, retmode='xml')
        records = Entrez.read(handle)
        
        for record in records['PubmedArticle']:
            title = record['MedlineCitation']['Article']['ArticleTitle']
            abstract = ' '.join(record['MedlineCitation']['Article']['Abstract']['AbstractText']) if 'Abstract' in record['MedlineCitation']['Article'] and 'AbstractText' in record['MedlineCitation']['Article']['Abstract'] else ''
            journal = record['MedlineCitation']['Article']['Journal']['Title']
            url = f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
            
            pmcid, conclusions = await self.fetch_pmcid_and_conclusions(pmid, session)
            
            return {
                'PMID': pmid,
                'Title': title,
                'Abstract': abstract,
                'Journal': journal,
                'URL': url,
                'PMCID': pmcid,
                'Conclusions': conclusions
            }

    async def scrape(self, full_query):
        async with ClientSession() as session:
            handle = Entrez.esearch(db='pubmed', retmax=11, term=full_query)
            record = Entrez.read(handle)
            id_list = record['IdList']

            results = []
            for pmid in id_list:
                result = await self.fetch_pubmed_record(pmid, session)
                if result is not None:  # Only add non-None results
                    results.append(result)
                await asyncio.sleep(1)  # Add a 1-second delay between requests
                print(f"Processed PMID: {pmid}")
            
            if not results:  # Check if results list is empty
                print("No valid results found.")
                return pd.DataFrame()  # Return an empty DataFrame
            
            return pd.DataFrame(results)

    def run(self, topics, date_range):
        full_query = self.build_query(topics, date_range)
        df = asyncio.run(self.scrape(full_query))
        if df.empty:
            print("No results were found for the given query.")
        return df