from search_util import LIMIT, load_movies , load_stopwords
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def search_command(args : str ,limit: int = LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = stem_tokens(args)
        title_tokens = stem_tokens(movie["title"])
        if matching_tokens(query_tokens,title_tokens):
            results.append(movie) 
            if(len(results) > limit):
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
    for unstemmed_token in  unstemmed_tokens:
        stemmed_tokens.append(stemmer.stem(unstemmed_token))
    return stemmed_tokens
    

    

def matching_tokens(str1 : list[str] , str2 : list[str]) -> bool:
    for s1 in str1:
        for s2 in str2:
            if s1 in s2:
                return True
    return False



    

