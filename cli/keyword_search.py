from search_util import LIMIT, load_movies , load_stopwords , DATA_PATH_CACHE
import string
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
import math

stemmer = PorterStemmer()

import pickle
import os

class inverted_index:
    def __init__(self):
        self.index = {} #index to document number
        self.docmap = {} # document number to the document itself
        self.term_frequencies = defaultdict(Counter) #document IDs to Counter objects
        self.index_file = os.path.join(DATA_PATH_CACHE, "index.pkl")
        self.docmap_file = os.path.join(DATA_PATH_CACHE, "docmap.pkl")
        self.term_frequencies_file = os.path.join(DATA_PATH_CACHE, "term_frequencies.pkl")
       
    
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
           self.index.setdefault(token, set()).add(doc_id)
        for token in tokens:
           self.term_frequencies[doc_id][token] += 1
        
    def get_documents(self, term) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id, term) -> int:
        token = tokenize_text(term)
        if len(token) > 1:
           raise ValueError("more than one token")
        token_str = token[0]
        return self.term_frequencies[doc_id][token_str]
    
    def get_idf(self , term : str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"IDF expects a single word, got: {tokens}")
                
        target_token = tokens[0]
        total_doc_count = len(self.docmap)
        term_match_docs = self.get_documents(target_token)
        term_match_doc_count = len(term_match_docs)
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
        
        
        

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

        with open(self.term_frequencies_file, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        
    
    def load(self):
        with open(self.index_file, 'rb') as f:
            self.index= pickle.load(f)
            
        with open(self.docmap_file, 'rb') as f:
            self.docmap= pickle.load(f)

        with open(self.term_frequencies_file , 'rb') as f:
            self.term_frequencies = pickle.load(f)
       
            
idx = inverted_index()
def build_command() -> None:
    
    idx.build()
    idx.save()

def search_command(args : str ,limit: int = LIMIT):
    try:
        idx.load()
                
    except Exception as e:
        print(e)
        exit(1)

    query_tokens = tokenize_text(args)
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
        results_title_desc.append(idx.docmap[doc_num])
    results = []
    for result_dict in results_title_desc:
        results.append(result_dict['title'])
    return results

def tokenize_text (text : str ) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    #filter out the stop words
    reduced_tokens = text.split()
    stop_words_list = load_stopwords()
    final_tokens = []
    for token in reduced_tokens:
        if token not in stop_words_list:
            final_tokens.append(token)
    # build the stemmed tokens
    unstemmed_tokens = final_tokens
    stemmed_tokens = []
    for unstemmed_token in unstemmed_tokens:
        stemmed_tokens.append(stemmer.stem(unstemmed_token))
    return stemmed_tokens

def tf_command(doc_id: int, term: str) -> int:
    idx = inverted_index()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = inverted_index()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = inverted_index()
    idx.load()
    return idx.get_tf_idf(doc_id, term)




    
