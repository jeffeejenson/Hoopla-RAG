from sentence_transformers import SentenceTransformer
import numpy as np
from search_util import load_movies,DATA_PATH_CACHE
import os


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







        
        