from keyword_search import tokeniser,pre_process
from search_util import load_movies,DATA_PATH_CACHE
import pickle
import os

class inverted_index:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.index_file = os.path.join(DATA_PATH_CACHE, "index.pkl")
        self.docmap_file = os.path.join(DATA_PATH_CACHE, "docmap.pkl")
    
    def __add_document(self, doc_id, text):
        tokens = tokeniser(text)
        for token in set(tokens):
           self.index.setdefault(token, set()).add(doc_id)
        
    def get_documents(self, term) -> list:
        token = pre_process(term)
        doc_list = list(self.index[token])
        doc_list.sort()
        return doc_list

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie['id'],f"{movie['title']}  {movie['description']}")
            self.docmap[movie['id']] = movie
    
    def save(self):
        os.makedirs(DATA_PATH_CACHE, exist_ok=True)
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.index, f)

        with open(self.docmap_file, 'wb') as d:
            pickle.dump(self.docmap, d)
    