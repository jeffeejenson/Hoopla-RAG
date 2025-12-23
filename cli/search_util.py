import json
import os
import stat

LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH_MOVIE = os.path.join(PROJECT_ROOT , "data" , "movies.json")
DATA_PATH_STOP_WORDS = os.path.join(PROJECT_ROOT , "data" , "stopwords.txt")
DATA_PATH_CACHE = os.path.join(PROJECT_ROOT,"cache")



def load_movies() -> list[dict]:
    with open(DATA_PATH_MOVIE, 'r') as file:
        data = json.load(file)
        movies_list = data["movies"]
        return movies_list

def load_stopwords() -> list[str]:
    with open(DATA_PATH_STOP_WORDS,'r') as file:
        stop_words = file.read()
        stop_words_list = stop_words.splitlines()
    return stop_words_list
"""
def path_create(root: str, sub: str) -> str:
    FILE_PATH = os.path.join(root, sub)
    if not os.path.exists(FILE_PATH):
        os.makedirs(FILE_PATH, exist_ok=True) 
    return FILE_PATH 
DATA_PATH_CACHE = path_create(PROJECT_ROOT, "cache")

"""

