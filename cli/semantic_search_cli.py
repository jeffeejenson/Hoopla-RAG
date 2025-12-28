#!/usr/bin/env python3

import argparse
from semantic_search import SemanticSearch,embed_text,verify_model,verify_embeddings,embed_query_text,embedded_search,chunk_text
from search_util import LIMIT,load_movies
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit" , type=int , default=LIMIT, help = "limit value")

    chunk_parser = subparsers.add_parser("chunk" ,  help="embedd movies as chunks")
    chunk_parser.add_argument("query" , type = str , help = "query to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, dest="chunk_size", help="The number of words or tokens to group into a single chunk. (Default: 200)")
    chunk_parser.add_argument("--overlap", type=int, dest="overlap", help="The number of overlap")





    


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
            embedded_search(args.query, limit=args.limit)
        
        case "chunk":
            chunk_text(args.query , args.chunk_size , args.overlap)


        case _:
            parser.print_help()

        


if __name__ == "__main__":
    main()