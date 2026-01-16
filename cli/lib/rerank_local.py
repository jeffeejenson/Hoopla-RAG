import os
import ollama
from sentence_transformers import CrossEncoder
import json





def result_rerank_individual_local(query : str , doc : dict) -> str:
    text = f"""Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {doc.get("title", "")} - {doc.get("description", "")}

                Consider:
                - Please Direct relevance to query
                - Consider User intent (what they're looking for)
                - Content appropriateness

                Eg. Query : "family movie about bears in the woods"
                    Movie : The Berenstain Bears' Christmas Tree - ranked 10 - perfect match

                Rate 0-10 (10 = perfect match).
                Give me ONLY the number in your response, no other text or explanation.


                Score:"""
    
    response = ollama.generate(model='gemma3:4b', prompt=text)
    score = float(response['response'])
    return score

def result_rerank_batch_local(query : str , doc_list_str : str) -> list:
    text = f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                [75, 12, 34, 2, 1]
            """
    response = ollama.generate(model='gemma3:4b', prompt=text)
    id_list = json.loads(response['response'])
    return id_list


def cross_encoder_local( query : str , results : list[dict] , limit : int) -> list[dict]:
    pairs = []

    for result in results:
        pairs.append([query, f"{result.get('title', '')} - {result.get('description', '')}"])

                
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    for score,result in zip(scores,results):
        result["Cross Encoder Score"] = score
                
                
    results = sorted(results , key = lambda item :item["Cross Encoder Score"] , reverse = True)[:limit]
    return results

    

    
                
