from claims.prompts import *
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from claims import logging

class QueryResponse(BaseModel):
    """
    A pydantic query response model for structuring the output response for printing.
    """
    scientific_validation_summary: str = Field(description=scientific_validation_summary_task)
    classification: str = Field(description=classification_task)
    research_summary: str = Field(description=research_summary_task)
    contradictory_claims: str = Field(description=contradictory_claims_task)


class RAGQueryProcessor:
    """
    A class to process queries using RAG and GPT-4o-mini.
    """
    def __init__(self, custom_retriever, gpt_prompt_txt):
        self.custom_retriever = custom_retriever
        self.gpt_prompt_txt = gpt_prompt_txt
        self.setup_components()

    #Initialize the components in the QueryProcessor class object
    def setup_components(self):

        try:
        
            self.parser= JsonOutputParser(pydantic_object= QueryResponse) #json output parser derived from the Query response object

            #prompt template with claim as input, gpt_prompt_text as prompt and formatting instructions using json parser
            self.gpt_prompt= PromptTemplate(
                template= self.gpt_prompt_txt,
                input_variables= ["claim"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )

            self.chatgpt= ChatOpenAI(model_name= 'gpt-4o-mini', temperature=0)

            self.setup_rag_chain() #initializing the retrievalQA rag chain

        except Exception as e:
            logging.info(f"Error setting up components: {e}")
            return None

    def format_docs(self, documents):
        return "\n\n".join(doc.page_content for doc in documents)

    def setup_rag_chain(self):
        # Set up the RetrievalQA chain
        self.retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=self.chatgpt,
        chain_type="stuff",
        retriever=self.custom_retriever,
        chain_type_kwargs={"prompt": self.gpt_prompt}
        )

        # Set up the runnable RAG chain
        self.retrieval_qa_rag_chain = (
            {
                "context": (self.custom_retriever | self.format_docs),
                "claim": RunnablePassthrough()
            }
            | self.gpt_prompt
            | self.chatgpt
        )
        
    def process_query_retrieval_qa(self, claim):
        """
        Process a query using the retrievalQA chain and GPT-4o-mini to generate a response.
        """
        try:        
            response= self.retrieval_qa_rag_chain.invoke(claim)
            return response.content
        except Exception as e:
            logging.info(f"Error processing query and returning generated text: {e}")
            return None


