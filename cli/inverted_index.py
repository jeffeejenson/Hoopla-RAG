from keyword_search import tokeniser
from search_util import load_movies,DATA_PATH_CACHE
import pickle
import os

class inverted_index:
    def __init__(self):
        self.index = {} #index to document number
        self.docmap = {} # document number to the document itself
        self.index_file = os.path.join(DATA_PATH_CACHE, "index.pkl")
        self.docmap_file = os.path.join(DATA_PATH_CACHE, "docmap.pkl")
        self.index_loaded = {}
        self.docmap_loaded = {}
    
    def __add_document(self, doc_id, text):
        tokens = tokeniser(text)
        for token in set(tokens):
           self.index.setdefault(token, set()).add(doc_id)
        
    def get_documents(self, term) -> list[int]:
        doc_ids = self.index_loaded.get(term, set())
        return sorted(list(doc_ids))

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie['id'],f"{movie['title']}  {movie['description']}")
            self.docmap[movie['id']] = movie
    
    def save(self) -> None:
        os.makedirs(DATA_PATH_CACHE, exist_ok=True)
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.index, f)

        with open(self.docmap_file, 'wb') as f:
            pickle.dump(self.docmap, f)
    
    def load(self):
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Index file not found: {self.index_file}")

        with open(self.index_file, 'rb') as f:
            self.index_loaded = pickle.load(f)

        if not os.path.exists(self.docmap_file):
            raise FileNotFoundError(f"Docmap file not found: {self.docmap_file}")

        with open(self.docmap_file, 'rb') as f:
            self.docmap_loaded = pickle.load(f)
       
            



    