

YoutubeSummary_task="""You are a Youtube video summarizer. You will be taking the trascript text and summarizing the entire video and give a useful summary which gives an entire picture/idea to the user
about the video. Please provide the summary of the text given here : """

ClaimGenerator_taks='''You are a Youtube Claims generator. You will be provided a summary of a youtube video in detail, especially a health and fitness related content of a youtube video. You have to
generate top 5 claims for our health claims verification project. The claims needs to be given in single line points separated by * and in proper order with proper and legit medical terminology. Structure the sentences as Subject-verb-object (SVO) format. Please provide the claims for the text
given here :'''

# Define your desired data structure - like a python data class.
scientific_validation_summary_task="""Provide scientific Validation summary in less than 25 words:**
   - Conduct a thorough review of the retrieved context for studies related to the provided claim.
   - Prioritize peer-reviewed journals, with special emphasis on systematic reviews, cohort studies, meta-analyses and randomized controlled trials (RCTs), if available in the context as they are high quality scientific evidence.
   - Do not consider case reports, case series, opinion pieces or observational studies and do not make up research papers as they are low quality evidence.
   - Evaluate the strength of evidence supporting the claim, as well as any contradictory or inconclusive findings.
   - If no relevant content available in the provided only then conduct a thorough review of reputable medical research databases like PubMed.
   """

classification_task= """Based on the context used to summarize in the above task, classify the claim as one of the following:
**Scientific**: Supported by substantial, high-quality scientific evidence.
**Pseudo-science/Inconclusive**: Not supported by strong and credible evidence OR supported only by inconclusive scientific evidence, or contradicted by substantial evidence.
**Partially correct**: Supported by substantial scientific evidence but with significant caveats."""

research_summary_task= """Research Summary in less than 25 words:Provide a concise summary of the research findings that support your classification."""

contradictory_claims_task= """Contradictory Claims in less than 25 words: Identify if there are any scientifically supported evidence that contradicts the original claim or pose any health risks.
If such evidence is found, explain why the contradicting claim is scientifically valid."""

gpt_prompt_txt= """
You are a medical researcher.Given the following health-related claim, generate the response based on the tasks specified in the following instructions:
claim= {claim}
context= {context}
Format Instructions: {format_instructions}
"""