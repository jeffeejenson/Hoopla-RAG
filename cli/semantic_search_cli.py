#!/usr/bin/env python3

import argparse
from semantic_search import SemanticSearch,embed_text,verify_model,verify_embeddings,embed_query_text
from search_util import LIMIT,load_movies
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit" , type=int , default=LIMIT, help = "limit value")
    


    verify = subparsers.add_parser("verify", help="verify build tool")

    embed_parser = subparsers.add_parser("embed_text", help="embed the text")
    embed_parser.add_argument("query", type=str, help="query term")

    embedquery = subparsers.add_parser("embedquery", help="embed the query")
    embedquery.add_argument("query", type=str, help="query term")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verify embedding tool")

    args = parser.parse_args()

   

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            embed_text(args.query)
        
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)

        case "search":
            semSearch = SemanticSearch()
            
          
            movies = load_movies()
            semSearch.load_or_create_embeddings(movies)
            
            
            results = semSearch.search(args.query, limit=args.limit)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}")
                print() # Add a newline between results for cleaner formatting
                
        case _:
            parser.print_help()

        


if __name__ == "__main__":
    main()