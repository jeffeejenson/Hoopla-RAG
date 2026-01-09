import argparse
import os
from hybrid_search import normalise_scores,HybridSearch
from search_util import load_movies
from dotenv import load_dotenv
from google import genai



def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalise_parser = subparsers.add_parser("normalize" , help = "normalise scores")
    normalise_parser.add_argument("scores" , nargs='+', help = "Provide the scores")

    weighted_search_parser = subparsers.add_parser("weighted-search", help = "perform weighted search")
    weighted_search_parser.add_argument("query" , type = str , help = "type in the query")
    weighted_search_parser.add_argument("--alpha" , type = float , default=0.5, help = "key in the alpha value")
    weighted_search_parser.add_argument("--limit" , type = int, default=5, help = "key in the limit" )

    rrf_search_parser = subparsers.add_parser("rrf-search" , help = "perfirm rrf search" )
    rrf_search_parser.add_argument("query" , type = str , help = "Type in the qquery")
    rrf_search_parser.add_argument("-k" , type = int ,default= 60, help = "K value for rrf")
    rrf_search_parser.add_argument("--limit" , type = int ,default= 5, help = "limit of movies")
    rrf_search_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="Query enhancement method",)
    rrf_search_parser.add_argument("--rerank-method" , type = str ,choices =["individual"], help = "rerank method")





    args = parser.parse_args()

    match args.command:
        case "normalize":
            mp_obj = map(float,args.scores)
            plain_scores = list(mp_obj)
            scores = normalise_scores(plain_scores)
            for score in scores:
                print(f"* {score:.4f}")

            
        case "weighted-search":
            hyb = HybridSearch(load_movies())
           # hyb.documents = load_movies()
            results = hyb.weighted_search(args.query,args.alpha,args.limit)

            for i,result in enumerate(results,1):
                print(f"{i}. {result["title"]} ")
                print(f"Hybrid Score :{result["Hybrid Score"]:.4f}")
                print(f"BM25 : {result["BM25"]:.4f}   ,Semantic : {result["Semantic"]:.4f}")
                print(f"{result["Description"]} ")
        
        case "rrf-search":

            hyb = HybridSearch(load_movies())
            results = []
            if args.enhance == None:
                results = hyb.rrf_search(args.query, args.k, args.limit)
            elif args.enhance == "spell":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                prompt = f"""Fix any spelling errors in this movie search query.

                            Only correct obvious typos. Don't change correctly spelled words.

                            Query: "{args.query}"

                            If no errors, return the original query.
                            Corrected:"""
                response = client.models.generate_content( model='gemini-2.5-flash-lite', contents=prompt)
                query = response.text
                results = hyb.rrf_search(query, args.k, args.limit)
                #'gemini-2.5-flash-lite'
                #'gemini-2.0-flash-001'
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            
            
            elif args.enhance == "rewrite":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                prompt = f"""Rewrite this movie search query to be more specific and searchable.

                            Original: "{args.query}"

                            Consider:
                                - Common movie knowledge (famous actors, popular films)
                                - Genre conventions (horror = scary, animation = cartoon)
                                - Keep it concise (under 10 words)
                                - It should be a google style search query that's very specific
                                - Don't use boolean logic

                            Examples:

                            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                            -  "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                            Rewritten query:"""
                response = client.models.generate_content( model='gemini-2.5-flash-lite', contents=prompt)
                query = response.text
                results = hyb.rrf_search(query, args.k, args.limit)
                #'gemini-2.5-flash-lite'
                #'gemini-2.0-flash-001'
            
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

            elif args.enhance == "expand":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                prompt =f"""Expand this movie search query with related terms.

                            Add synonyms and related concepts that might appear in movie descriptions.
                            Keep expansions relevant and focused.
                            This will be appended to the original query.

                            Examples:

                            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                            - "action movie with bear" -> "action thriller bear chase fight adventure"
                            - "comedy with bear" -> "comedy funny bear humor lighthearted"

                            Query: "{args.query}"""
                response = client.models.generate_content( model='gemini-2.5-flash-lite', contents=prompt)
                query = response.text
                results = hyb.rrf_search(query, args.k, args.limit)
                #'gemini-2.5-flash-lite'
                #'gemini-2.0-flash-001'
            
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            
            if args.rerank_method == "individual":




                            
            for i,result in enumerate(results,1):
                print(f"{i}. {result["title"]} ")
                print(f"RRF Score :{result["rrf_score"]:.4f}")
                print(f"BM25 RANK : {result["bm25_rank"]:.4f}   ,Semantic Rank : {result["sem_rank"]:.4f}")
                print(f"{result["description"][:100]} ")




        case _:
            parser.print_help()


if __name__ == "__main__":
    main()