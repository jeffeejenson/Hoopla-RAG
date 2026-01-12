from dotenv import load_dotenv
from google import genai
from search_util import load_movies
import os
import time
from hybrid_search import normalise_scores,HybridSearch


#'gemini-2.5-flash-lite'
#'gemini-2.0-flash-001'



def result_rerank_individual(query : str , doc : dict) -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("description", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Give me ONLY the number in your response, no other text or explanation.

                Score:"""
    
    response = client.models.generate_content( model='gemini-2.5-flash-lite', contents=prompt)
    return response.text

def result_rerank_batch(query : str , doc_list_str : list[str]):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                [75, 12, 34, 2, 1]
            """
    response = client.models.generate_content( model='gemini-2.5-flash-lite', contents=prompt)
    return response.text
    

    
                
