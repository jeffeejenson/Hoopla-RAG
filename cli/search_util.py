import json
import os

LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH_MOVIE = os.path.join(PROJECT_ROOT , "data" , "movies.json")
DATA_PATH_STOP_WORDS = os.path.join(PROJECT_ROOT , "data" , "stopwords.txt")



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
        
