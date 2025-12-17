from search_util import LIMIT, load_movies , load_stopwords , DATA_PATH_CACHE
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

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
        tokens = stem_tokens(text)
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
       
            
idx = inverted_index()
def build_command() -> None:
    
    idx.build()
    idx.save()

def search_com(args : str ,limit: int = LIMIT):
    
    try:
        idx.load()
                
    except Exception as e:
        print(e)
        exit(1)

    query_tokens = stem_tokens(args)
    doc_set = set()
    for query_token in query_tokens:
        documents = idx.get_documents(query_token)
        for document in documents:
            doc_set.add(document)
    doc_num_list = list(doc_set)
    doc_num_list.sort()
    doc_list = doc_num_list[0:limit]

    results_title_desc = []
    for doc_num in doc_list:
        results_title_desc.append(idx.docmap_loaded[doc_num])
    results = []
    for result_dict in results_title_desc:
        results.append(result_dict['title'])
    return results

    


def search_command(args : str ,limit: int = LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    query_tokens = stem_tokens(args)
    for movie in movies:
        title_tokens = stem_tokens(movie["title"])
        if matching_tokens(query_tokens,title_tokens):
            results.append(movie) 
            if(len(results) >= limit):
                break
    results.sort(key = lambda movie : movie["id"])
    return results    
    
def pre_process( text : str ) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokeniser ( text : str) -> list[str]:
    text = pre_process(text)
    return text.split()

def stop_words(text : str) -> list[str]:
    reduced_tokens = tokeniser(text)
    stop_words_list = load_stopwords()
    final_tokens = []
    for token in reduced_tokens:
        if token not in stop_words_list:
            final_tokens.append(token)
    return final_tokens

def stem_tokens(text : str) -> list[str]:
    unstemmed_tokens = stop_words(text)
    stemmed_tokens = []
    for unstemmed_token in unstemmed_tokens:
        stemmed_tokens.append(stemmer.stem(unstemmed_token))
    return stemmed_tokens
    
def matching_tokens(str1 : list[str] , str2 : list[str]) -> bool:
    for s1 in str1:
        for s2 in str2:
            if s1 in s2:
                return True
    return False



    

