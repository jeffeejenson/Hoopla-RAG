from search_util import GOLDEN_DATA_SET
from hybrid_search import rrf_search_command
import json
import ollama



def precision_k(limit : int) -> list[dict]:

    with open(GOLDEN_DATA_SET, "r") as f:
        golden_dataset = json.load(f)

    
    scored_results : list[dict] = []

    for test_case in golden_dataset["test_cases"]:
        results = rrf_search_command(test_case["query"] , 60 , limit)
        test_titles : list[str] = []
        title_results : list[str] = []
        for test_title in test_case["relevant_docs"]:
            test_titles.append(test_title)
        for result in results:
            title_results.append(result["title"])

        relevant_docs = 0
        for title_result in title_results:
            for test_title in test_titles:
                if title_result == test_title:
                    relevant_docs += 1

        retrived_docs = len(title_results)
        precision_k_score = float(relevant_docs / retrived_docs)

        #recall = float(relevant_docs / len(test_titles))
        test_case
        recall = float(relevant_docs / len(test_case["relevant_docs"]))

    

        f1 = float(2 * (precision_k_score * recall) / (precision_k_score + recall)) if precision_k_score + recall > 0 else 0.0
        
        #print(test_case["query"])

        #print("total relevant:", len(test_titles))
        #print("relevant in top k:", relevant_docs)
        #print("retrieved:", title_results)
        #print("relevant:", test_titles)
        
        scored_result : dict = {"query" : test_case['query'] ,"precision" : precision_k_score,"recall" : recall ,"f1": f1,"retrieved_docs": title_results,"relevant_docs": test_titles }
        scored_results.append(scored_result)
    
    return scored_results
        
def precision_k1(limit : int) -> list[dict]:

    with open(GOLDEN_DATA_SET, "r") as f:
        golden_dataset = json.load(f)

    
    scored_results : list[dict] = []

    for test_case in golden_dataset["test_cases"]:
        results = rrf_search_command(test_case["query"] , 60 , limit)
        test_titles = set()
        title_results = set()
        for test_title in test_case["relevant_docs"]:
            test_titles.add(test_title)
        for result in results:
            title_results.add(result["title"])

        relevant_docs = len(test_titles & title_results)

        retrived_docs = len(title_results)
        precision_k_score = float(relevant_docs / retrived_docs)

        recall = float(relevant_docs / len(test_titles))
    

        f1 = float(2 * (precision_k_score * recall) / (precision_k_score + recall)) if precision_k_score + recall > 0 else 0.0
        # "The Fast and the Furious",
        f1 = 0.0
        scored_result : dict = {"query" : test_case['query'] ,"precision" : precision_k_score,"recall" : recall ,"f1": f1,"retrieved_docs": list(title_results),"relevant_docs": list(test_titles) }
        scored_results.append(scored_result)
    
    return scored_results

def llm_judge_results(query: str, results: list[dict]) -> list[int]:
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result['title']}")

    text = f"""### INSTRUCTION
Rate the relevance of the following Results to the Query using a scale of 0-3.
Scale:
3 = Highly relevant
2 = Relevant
1 = Marginally relevant
0 = Not relevant

### CONSTRAINTS
- Output MUST be a valid JSON array of integers.
- Do NOT include any text, preamble, or markdown code blocks (```).
- Output ONLY the numbers.
- Maintain the exact order of the Results provided.

### DATA
Query: "{query}"

Results:
{chr(10).join(formatted_results)}

### FINAL RESPONSE FORMAT
[int, int, int, ...]

### OUTPUT"""
    
    response = ollama.generate(model='gemma3:4b', prompt=text)

    ranking_text = (response['response'] or "").strip()
    scores = json.loads(ranking_text)

    if len(scores) == len(results):
        return list(map(int, scores))

    raise ValueError(
        f"LLM response parsing error. Expected {len(results)} scores, got {len(scores)}. Response: {scores}"
    )





















