from sentence_transformers import SentenceTransformer
import numpy as np
from search_util import load_movies,DATA_PATH_CACHE,LIMIT,DEFAULT_CHUNK_OVERLAP,DEFAULT_CHUNK_SIZE,DEFAULT_SEMANTIC_CHUNK_SIZE,SCORE_PRECISION
import os
import re
import json


class SemanticSearch :
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.documents_map = {}
        self.embedding_path = os.path.join(DATA_PATH_CACHE, "movie_embeddings.npy")

    def build_embeddings(self, documents : list[dict]):
        self.documents = documents
        movie_title_desc = []
        
        for document in documents:
            self.documents_map[document["id"]] = document
            movie_title_desc.append(f"{document['title']}  {document['description']}")

        self.embeddings = self.model.encode(movie_title_desc, show_progress_bar = True)
        np.save(self.embedding_path, self.embeddings, allow_pickle=True)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}

        
        for document in documents:
            self.documents_map[document["id"]] = document

        if os.path.exists(self.embedding_path):
            self.embeddings = np.load(self.embedding_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        
        return self.build_embeddings(documents)


    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError ("text is empty")
        #return self.model.encode([text])[0]
        embedding = []
        embedding = self.model.encode([text])
        return embedding[0]
    
    def search(self, query, limit = 5):
        if self.embeddings is None:
            raise ValueError ("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        top_matches = []
        for doc_embedding,document in zip(self.embeddings,self.documents):
            sim_score = cosine_similarity(query_embedding,doc_embedding)
            top_matches.append((sim_score,document))
        top_matches_sorted = sorted(top_matches, key=lambda item: item[0], reverse=True)[:limit]
        results = []
        for top_match_sorted in top_matches_sorted:
            temp_dict = { "score" : top_match_sorted[0] , "title" : top_match_sorted[1]["title"] , "description" : top_match_sorted[1]["description"] }
            results.append(temp_dict)
        return results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        
        for document in documents:
            self.documents_map[document["id"]] = document

        chunks = [] #list[str]
        meta_data = [] # list[dict]

        for i,document in enumerate(documents,1):
            if document["description"] != None:
                chunk = semantic_chunk1(document["description"], 4 ,1)
                total_chunks = len(chunk)
                for ch_idx,ch in enumerate(chunk):
                    chunks.append(ch)
                    temp_dict = { 'movie_idx' : document["id"] , 'chunk_idx' : ch_idx,'total_chunks' : total_chunks}
                    meta_data.append(temp_dict)
                

            
        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = meta_data
        np.save("cache/chunk_embeddings.npy",self.chunk_embeddings)

        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2 )

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        
        for document in documents:
            self.documents_map[document["id"]] = document
        
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            with open("cache/chunk_embeddings.npy", "rb") as f:
                self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json" , "rb") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings
        

        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = super().generate_embedding(query)
        chunk_scores = []
        self.chunk_embeddings = self.load_or_create_chunk_embeddings(load_movies())
        for ch_idx ,(chunk_embedding,movie_chunk_metadata) in enumerate(zip(self.chunk_embeddings,self.chunk_metadata)):
            score = cosine_similarity(query_embedding,chunk_embedding)
            temp_dict = { 'chunk_idx' : ch_idx, 'movie_idx' : movie_chunk_metadata["movie_idx"] ,'score' : score}
            chunk_scores.append(temp_dict)

        movies_scores = {} #movie indexes to scores
        for chunk_score in chunk_scores:
            if chunk_score["movie_idx"] not in movies_scores or chunk_score["score"] >= movies_scores.get(chunk_score["movie_idx"], -1):
                movies_scores[chunk_score["movie_idx"]] = chunk_score["score"]
        movies_scores_sorted = sorted(movies_scores.items(),key=lambda item: item[1], reverse=True )
        movies_scores_sorted_limited = movies_scores_sorted[:limit]

        results = []

        for m,meta in zip(movies_scores_sorted_limited, self.chunk_metadata):
            doc_id = m[0]
            title = self.documents_map[doc_id]["title"]
            document = self.documents_map[doc_id]["description"]
            score = m[1]
            metadata = meta
            dict_temp = { "id": doc_id, "title": title, "document": document[:100], "score": round(score, SCORE_PRECISION),"metadata": metadata or {}}
            results.append(dict_temp)

        return results


def verify_model():
    s = SemanticSearch()
    print("Model loaded: {s.model}")
    MAX_LENGTH = s.model.max_seq_length
    print(f"Max sequence length: {MAX_LENGTH}")

def embed_text(text : str):
    semSearch = SemanticSearch()
    embedding = semSearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")  

def verify_embeddings():
    semSearch = SemanticSearch()
    movies = load_movies()
    embeddings = semSearch.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(semSearch.documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semSearch = SemanticSearch()
    embedding = semSearch.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embedded_search_func(query : str, limit= LIMIT) -> list[dict]:
    semSearch = SemanticSearch()      
    movies = load_movies()
    semSearch.load_or_create_embeddings(movies)        
    results = semSearch.search(query, limit)
    return results
    
def embedded_search(query : str, limit= LIMIT) -> None:      
    results = embedded_search_func(query, limit)
            
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description']}")
        print() 

def chunk_text_func( query : str , chunk_size :int ,overlap :int) -> list[str]:
    words = query.split()
    chunks = []
    overlap_words = []
        
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk_sentence = []
        if overlap_words:
            chunk_words = overlap_words + chunk_words
        if overlap > 0:
                overlap_words = chunk_words[-overlap:] if len(chunk_words) >= overlap else chunk_words
        else:
            overlap_words = []

        chunk_sentence = " ".join(chunk_words)
        chunks.append(chunk_sentence)
    return chunks

def chunk_text( query : str , chunk_size :int ,overlap :int) -> None:
    chunks = chunk_text_func(query,chunk_size,overlap)
    count = len(query)
    print(f"Chunking {count} characters")

    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def chunk_text_semantic( query : str , chunk_size :int ,overlap :int) -> list[str]:
    

    words =re.split(r"(?<=[.!?])\s+", query)
    
    chunks = []
    overlap_words = []
        
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk_sentence = []
        if overlap_words:
            chunk_words = overlap_words + chunk_words
        if overlap > 0:
            overlap_words = chunk_words[-overlap:] if len(chunk_words) >= overlap else chunk_words
        else:
            overlap_words = []

        chunk_sentence = " ".join(chunk_words)
        chunks.append(chunk_sentence)
    
    return chunks

def semantic_chunk1(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    i = 0
    n_sentences = len(sentences)

    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            cleaned_sentences.append(chunk_sentence.strip())
        if not cleaned_sentences:
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap

    return chunks




    










        
        