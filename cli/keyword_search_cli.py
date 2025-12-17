#!/usr/bin/env python3

import argparse
from search_util import load_movies
from keyword_search import *
from inverted_index import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
   
    build_parser = subparsers.add_parser("build", help="Build the inverted index cache")

    args = parser.parse_args()
    
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_com(args.query)
            
            for i , result in enumerate(results,1):
                print(f"{i}.{result}")
        
        case "build":
            build_command()
            

        case _ :
            parser.print_help()



if __name__ == "__main__":
    main()