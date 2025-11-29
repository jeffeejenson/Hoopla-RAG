import json
import os
import stat

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

def path_create(root : str , sub : str ) -> str:
    FILE_PATH = os.path.join(root , sub)
    if os.path.exists(FILE_PATH):
        path_exists = FILE_PATH
        return path_exists
    else:
        new_path = os.mkdir(FILE_PATH)
        return new_path
    
    """
    full_path = os.path.join(root, sub)

    if os.path.exists(full_path):
        return full_path
    else:
        os.mkdir(full_path, mode=0o777)
        os.chmod(full_path, 0o777)
        
        return full_path
    """
    
  
        




