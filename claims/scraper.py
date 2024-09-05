import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from Bio import Entrez
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import logging

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
    def __init__(self, email, api_key, max_concurrent_requests=10):
        self.email = email
        self.api_key = api_key
        Entrez.email = email
        Entrez.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None
        self.requests_per_second = 10 if api_key else 3
        self.request_times = []
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    def build_query(self, topics, date_range):
        queries = ['{}[Title/Abstract]'.format(topic) for topic in topics]
        full_query = f"({' AND '.join(queries)}) AND {date_range}"
        return full_query

    async def rate_limit(self):
        async with self.lock:
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 1]
            if len(self.request_times) >= self.requests_per_second:
                sleep_time = 1 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            self.request_times.append(time.time())

    async def fetch_with_rate_limit(self, url):
        await self.rate_limit()
        async with self.session.get(url) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")
            return await response.text()

    async def fetch_with_semaphore(self, url):
        async with self.semaphore:
            return await self.fetch_with_rate_limit(url)

    async def fetch_pmc_conclusions(self, pmcid):
        efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml&api_key={self.api_key}"

        try:
            content = await self.fetch_with_semaphore(efetch_url)
            root = ET.fromstring(content)

            conclusions = ""
            for section in root.findall(".//sec"):
                section_title = section.find("title")
                if section_title is not None and "conclusion" in section_title.text.lower():
                    conclusions = " ".join(p.text for p in section.findall(".//p") if p.text)
                    break

            return conclusions.strip() if conclusions else "Conclusions section not found."
        except Exception as e:
            logging.error(f"Error fetching PMC content for {pmcid}: {str(e)}")
            return f"Failed to fetch PMC content: {str(e)}"

    async def fetch_pmcid_and_conclusions(self, pmid):
        elink_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=pmc&id={pmid}&retmode=xml&api_key={self.api_key}"

        try:
            content = await self.fetch_with_semaphore(elink_url)
            root = ET.fromstring(content)
            pmcid_element = root.find(".//LinkSetDb/Link/Id")

            if pmcid_element is not None:
                pmcid = f"PMC{pmcid_element.text}"
                conclusions = await self.fetch_pmc_conclusions(pmcid)
                return pmcid, conclusions
            else:
                return None, "No PMC article available"
        except Exception as e:
            logging.error(f"Error fetching PMCID for {pmid}: {str(e)}")
            return None, f"Failed to fetch PMC article: {str(e)}"

    async def fetch_pubmed_record(self, pmid):
        await self.rate_limit()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._fetch_pubmed_record, pmid)
            return await asyncio.get_event_loop().run_in_executor(None, future.result)

    def _fetch_pubmed_record(self, pmid):
        try:
            handle = Entrez.efetch(db='pubmed', id=pmid, retmode='xml', api_key=self.api_key)
            records = Entrez.read(handle)

            for record in records['PubmedArticle']:
                article = record['MedlineCitation']['Article']
                return {
                    'PMID': pmid,
                    'Title': article['ArticleTitle'],
                    'Abstract': ' '.join(article['Abstract']['AbstractText']) if 'Abstract' in article and 'AbstractText' in article['Abstract'] else '',
                    'Journal': article['Journal']['Title'],
                    'URL': f"https://www.ncbi.nlm.nih.gov/pubmed/{pmid}"
                }
        except Exception as e:
            logging.error(f"Error fetching PubMed record for {pmid}: {str(e)}")
            return {'PMID': pmid, 'Error': str(e)}

    async def process_pmid(self, pmid):
        record = await self.fetch_pubmed_record(pmid)
        pmcid, conclusions = await self.fetch_pmcid_and_conclusions(pmid)
        record.update({'PMCID': pmcid, 'Conclusions': conclusions})
        return record

    async def scrape(self, full_query):
        await self.rate_limit()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._esearch, full_query)
            id_list = await asyncio.get_event_loop().run_in_executor(None, future.result)

        tasks = [self.process_pmid(pmid) for pmid in id_list]
        results = await asyncio.gather(*tasks)
        return pd.DataFrame(results)

    def _esearch(self, full_query):
        handle = Entrez.esearch(db='pubmed', retmax=10, term=full_query, api_key=self.api_key)
        record = Entrez.read(handle)
        return record['IdList']

    async def run(self, topics, date_range):
        full_query = self.build_query(topics, date_range)
        return await self.scrape(full_query)
