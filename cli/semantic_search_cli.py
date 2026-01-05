#!/usr/bin/env python3

import argparse
from semantic_search import SemanticSearch,embed_text,verify_model,verify_embeddings,embed_query_text,embedded_search,chunk_text,chunk_text_semantic,ChunkedSemanticSearch,semantic_chunk1
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

    semantic_chunk = subparsers.add_parser("semantic_chunk", help = "chunk text using semantics")
    semantic_chunk.add_argument("query" , type = str , help = "query to be chunked")
    semantic_chunk.add_argument("--max-chunk-size" , type=int ,default=4 ,  help = "max_chunk size to be chunked")
    semantic_chunk.add_argument("--overlap" ,type=int ,default=0 ,  help = "max_chunk size to be chunked")

    embed_chunk_parser = subparsers.add_parser("embed_chunks", help = "embed chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help = "search for semantic search, chunked")
    search_chunked_parser.add_argument("query" , type = str , help = "add query to search" )
    search_chunked_parser.add_argument("--limit" , type = int ,default=5, help = "add a limit")



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
        case "semantic_chunk":
            results = semantic_chunk1(args.query , args.max_chunk_size , args.overlap)
            for i,result in enumerate(results,1):
                print(f"{i}. {result}")

        case "embed_chunks":
            movies = load_movies()
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        
        case "search_chunked":
            movies = load_movies()
            css = ChunkedSemanticSearch()
            results = css.search_chunks(args.query , args.limit)

            for i,result in enumerate(results,1):
                TITLE = result["title"]
                SCORE = result["score"]
                DESCRIPTION = result ["document"]
                print(f"\n{i}. {TITLE} (score: {SCORE:.4f})")
                print(f"   {DESCRIPTION}...")

            


           

            

        case _:
            parser.print_help()

        


if __name__ == "__main__":
    main()