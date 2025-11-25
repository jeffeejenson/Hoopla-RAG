from search_util import LIMIT, load_movies
import string

def search_command(args : str ,limit: int = LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokeniser(args)
        title_tokens = tokeniser(movie["title"])
        """
        for args in args_query:
            if arg in title and len(results) < limit:
                if movie in results:
                    continue
                else:
                    results.append(movie) 
        #
                    if movie not in results else results.append("")
        """
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

def matching_tokens(str1 : list[str] , str2 : list[str]) -> bool:
    for s1 in str1:
        if s1 in str2:
            return True
    return False



    

