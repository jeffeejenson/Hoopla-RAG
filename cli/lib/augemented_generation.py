from hybrid_search import rrf_search_command
import ollama



def augement_result( query : str , results : list[dict]) -> str:
    formatted_results = [f"{i+1}. {d['title']} ({d['description']})" for i, d in enumerate(results)]
    final_string = "\n".join(formatted_results)

    text = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Documents:
                {final_string}

                Provide a comprehensive answer that addresses the query:"""
    response = ollama.generate(model='gemma3:4b', prompt=text)
    return response['response']


def rag_command( query : str ) -> dict:
    results = rrf_search_command(query , 60 , 5)
    rag_response = augement_result(query, results)

    final_result :dict = { "results" : results , "rag-response": rag_response}
    return final_result


def summarise_result(query : str , results : list[dict]) -> str:
    formatted_results = [f"{i+1}. {d['title']} ({d['description']})" for i, d in enumerate(results)]
    final_string = "\n".join(formatted_results)
    text = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{final_string}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    response = ollama.generate(model='gemma3:4b', prompt=text)
    return response['response']
    
    

def summarise_command(query : str , limit :int ):
    results = rrf_search_command(query , 60, limit)
    rag_response = summarise_result(query, results)
    final_result :dict = { "results" : results , "summarise-respone": rag_response}
    return final_result

def citation_result(query : str , results : list[dict]) -> str:
    formatted_results = [f"{i+1}. {d['title']} ({d['description']})" for i, d in enumerate(results)]
    final_string = "\n".join(formatted_results)
    text = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{final_string}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    response = ollama.generate(model='gemma3:4b', prompt=text)
    return response['response']

def citations_command(query : str , limit :int):
    results = rrf_search_command(query , 60, limit)
    rag_response = citation_result(query, results)
    final_result :dict = { "results" : results , "citation-respone": rag_response}
    return final_result

def question_result(query : str , results : list[dict]) -> str:
    formatted_results = [f"{i+1}. {d['title']} ({d['description']})" for i, d in enumerate(results)]
    final_string = "\n".join(formatted_results)
    text = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {query}

Documents:
{final_string}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation
- Dig into the description to find out answers

Answer:"""
    response = ollama.generate(model='gemma3:4b', prompt=text)
    return response['response']


def question_command(query : str , limit :int):
    results = rrf_search_command(query , 60, limit)
    rag_response = question_result(query, results)
    final_result :dict = { "results" : results , "question-respone": rag_response}
    return final_result


    


