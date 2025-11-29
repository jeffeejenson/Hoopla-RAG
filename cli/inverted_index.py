from keyword_search import tokeniser,pre_process
from search_util import load_movies,PROJECT_ROOT,path_create
import pickle
import os

class inverted_index:
    index = {}
    docmap = {}

    def __add_document(self, doc_id, text):
        tokens = tokeniser(text)
        for token in tokens:
           self.index.setdefault(token, set()).add(doc_id)
        
    def get_documents(self, term) -> list:
        token = pre_process(term)
        doc_list = list(self.index[token])
        doc_list.sort()
        return doc_list

    def build(self):
        movies = load_movies()
        for movie in movies:
            inverted_index.__add_document(self , movie['id'],f"{movie['title']}  {movie['description']}")
            self.docmap[movie['id']] = movie
    
    def save(self):
        DATA_PATH_CACHE = path_create(PROJECT_ROOT, "cache")
    
        # Create separate file paths for index and docmap
        index_file = os.path.join(DATA_PATH_CACHE, "index.pkl")
        docmap_file = os.path.join(DATA_PATH_CACHE, "docmap.pkl")
    
        with open(index_file, 'wb') as f:
            pickle.dump(self.index, f)

        with open(docmap_file, 'wb') as d:
            pickle.dump(self.docmap, d)


       


        





        











