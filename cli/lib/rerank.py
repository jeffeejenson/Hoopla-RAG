from dotenv import load_dotenv
import os
from google import genai
from sentence_transformers import CrossEncoder
import json





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

def result_rerank_batch(query : str , doc_list_str : str) -> list:
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
    id_list = json.loads(response.text)
    return id_list

def cross_encoder( query : str , results : list[dict] , limit : int) -> list[dict]:
    pairs = []

    for result in results:
        pairs.append([query, f"{result.get('title', '')} - {result.get('description', '')}"])

                
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    for score,result in zip(scores,results):
        result["Cross Encoder Score"] = score
                
                
    results = sorted(results , key = lambda item :item["Cross Encoder Score"] , reverse = True)[:limit]
    return results

    

    
                
