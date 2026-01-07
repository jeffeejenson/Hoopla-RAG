import os

from keyword_search import inverted_index,bm25_command
from semantic_search import ChunkedSemanticSearch


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
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
    def hybrid_score(self , bm25_score : float, semantic_score : float, alpha=0.5) -> float:
        return alpha * bm25_score + (1.0 - alpha) * semantic_score

def normalise_scores( score_element : float , scores : list[float]) -> float:
    if not scores == None:
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return 1.0000
        return (score_element - min_score) / (max_score - min_score)

            
        