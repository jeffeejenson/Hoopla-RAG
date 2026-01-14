import os
import time


from keyword_search import inverted_index,bm25_command
from semantic_search import ChunkedSemanticSearch
from search_util import load_movies
from enhance import spell_correction_query , rewrite_query , expand_query 
from rerank import result_rerank_individual , result_rerank_batch , cross_encoder
from enhance_local import spell_correction_query_local,rewrite_query_local,expand_query_local
from rerank_local import result_rerank_batch_local,result_rerank_individual_local,cross_encoder_local



class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = inverted_index()
        if not os.path.exists(self.idx.index_file):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        #self.idx.load()
        results = bm25_command(query)[:limit]
        return results

    def weighted_search(self, query, alpha, limit=5) -> list[dict]:
        bm25_results = self._bm25_search(query,limit * 500)
        bm25 = [] #list of dictinaries that maps from docID to bm25score(normalised)
        bm25_scores = [d['score'] for d in bm25_results]
        for bm25_result in bm25_results:
            temp_dict = {"docid" : bm25_result["docID"] , "score" : normalise_scores(bm25_result["score"],bm25_scores)}
            bm25.append(temp_dict)
    
        semantic_results = self.semantic_search.search_chunks(query , limit * 500)
        sem = [] #list of dictinaries that maps from docID to semantic chunk score(normalised)
        sem_scores = [d['score'] for d in semantic_results]
        for semantic_result in semantic_results:
            temp_dict = {"docid" : semantic_result["id"] , "score": normalise_scores(semantic_result["score"],sem_scores) }
            sem.append(temp_dict)

        #get unique set of document ids withinn keywrod results and semantic results
        docids = set()
        for b in bm25:
            docids.add(b["docid"])
        for s in sem:
            docids.add(s["docid"])
        
        doc_list = list(docids)

        unsorted_results = [] #this should have the list of dictinoartiers, documents iwth hybrid score, bm25score, semantic score
        bm25_map = {item['docid']: item['score'] for item in bm25}
        sem_map = {item['docid']: item['score'] for item in sem}
        for doc in doc_list:
            bm25_score = bm25_map.get(doc,0.0)
            sem_score = sem_map.get(doc,0.0)
            hybrid_score = self.hybrid_score( bm25_score, sem_score , alpha)
            temp_dict = {"title" : self.semantic_search.documents_map[doc]["title"] ,"Hybrid Score" : hybrid_score , "BM25" : bm25_score, "Semantic" : sem_score, "Description" : self.semantic_search.documents_map[doc]["description"]}
            unsorted_results.append(temp_dict)

        results = sorted(unsorted_results , key = lambda item: item['Hybrid Score'] , reverse = True) [:limit]
        
        return results    

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query,limit * 500)
        bm25 : list[dict] = []
        bm25_results = sorted(bm25_results , key = lambda item : item["score"] , reverse = True)
        for i,result in enumerate(bm25_results,1):
            temp_dict = {"docID" : result["docID"], "bm25_rank" : i, "bm25_score": result["score"]}
            bm25.append(temp_dict)
        

        sem_results = self.semantic_search.search_chunks(query , limit * 500)
        sem : list[dict] = []
        sem_results = sorted(sem_results , key =lambda item : item["score"] , reverse = True )
        for i,result in enumerate(sem_results , 1):
            temp_dict = {"docID" : result["id"], "sem_rank" : i, "sem_score": result["score"]}
            sem.append(temp_dict)
        
        combined : dict[int,dict] = {}

        for bm in bm25:
            bm["sem_rank"] = 0
            bm["sem_score"] = 0

            combined[bm["docID"]] = bm
        
        for sm in sem:
            if sm["docID"] in combined:
                combined[sm["docID"]]["sem_rank"] = sm["sem_rank"]
                combined[sm["docID"]]["sem_score"] = sm["sem_score"]
            else:
                sm["bm25_rank"] = 0
                sm["bm25_score"] = 0

                combined[sm["docID"]] = sm
        
        for doc_id in combined:
            rrf_bm25 = self.rrf_score(combined[doc_id]["bm25_rank"],k)
            rrf_sem = self.rrf_score(combined[doc_id]["sem_rank"],k)
            rrf_total = rrf_bm25 + rrf_sem

            combined[doc_id]["rrf_score"] = rrf_total

        results : list[dict] = []

        for doc_id,scores in combined.items():
            temp_dict = {"title" : self.semantic_search.documents_map[doc_id]["title"], "rrf_score": scores["rrf_score"] , "bm25_rank" : scores["bm25_rank"], "sem_rank" : scores["sem_rank"], "description":self.semantic_search.documents_map[doc_id]["description"] }
            results.append(temp_dict)

        final_results = sorted(results , key = lambda item :item["rrf_score"] , reverse = True)[:limit]

        return final_results

    def rrf_score(self , rank, k=60):
        if rank == 0:
            return 0
        return 1 / (k + rank)
    
    def hybrid_score(self , bm25_score : float, semantic_score : float, alpha=0.5) -> float:
        return alpha * bm25_score + (1.0 - alpha) * semantic_score

def normalise_scores( score_element : float , scores : list[float]) -> float:
    if not scores == None:
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return 1.0000
        return (score_element - min_score) / (max_score - min_score)
    
def normalise_command( scores : list[str] ) -> list[float]:
    mp_obj = map(float,scores)
    plain_scores = list(mp_obj)
    normalised_scores = []
    for each_score in plain_scores:
        normalised_scores.append(normalise_scores(each_score,plain_scores))
    
    return normalised_scores

def weighted_search_command( query : str , alpha : float , limit : int) -> list[dict]:
    hyb = HybridSearch(load_movies())
    results = hyb.weighted_search( query, alpha, limit)
    return results

def rrf_search_command(query : str , k : int , limit : int , enhance : str = None, rerank : str = None) -> list[dict]:
    hyb = HybridSearch(load_movies())
    results : list[dict] = []
    if enhance == None:
        results = hyb.rrf_search(query, k, limit)

    elif enhance == "spell":
        spell_corrected_query = spell_correction_query_local(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{spell_corrected_query}'\n")
        results = hyb.rrf_search(spell_corrected_query, k, limit )
        
    elif enhance == "rewrite":
        rewritten_query = rewrite_query_local(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{rewritten_query}'\n")
        results = hyb.rrf_search(rewritten_query , k , limit)
        
    elif enhance == "expand":
        expanded_query = expand_query_local(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{expanded_query}'\n")
        results = hyb.rrf_search(expanded_query , k , limit)
    
    if rerank == "individual":
        results = hyb.rrf_search(query, k, limit *5 )
        for result in results:
            result["reranked_score"] = float(result_rerank_individual_local(query , result))
            #time.sleep(20)
                
        results = sorted(results , key = lambda item :item["reranked_score"] , reverse = True)[:limit]

    if rerank == "batch":
        results = hyb.rrf_search(query, k, limit * 5)
        result_string : str = ""
                
        for i,result in enumerate(results,1):
            result["id"] = i
            result_string += str(result)

        id_list = result_rerank_batch_local(query , result_string)

        reranked_results :list[dict] = []

        for id in id_list:
            for result in results:
                if result["id"] == id:
                    reranked_results.append(result)
                
        results = reranked_results[:3]
    
    if rerank == "cross_encoder":
        results = cross_encoder_local(query , hyb.rrf_search(query , k , limit * 5) , limit)

    return results
    
    
    
    


    

            
        