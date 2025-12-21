#!/usr/bin/env python3

import argparse
from search_util import load_movies
from keyword_search import *
import math
#from sklearn.feature_extraction.text import TfidfVectorizer



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
   
    build_parser = subparsers.add_parser("build", help="Build the inverted index cache")
    tf = subparsers.add_parser("tf", help="Build term frequencies")
    tf.add_argument("doc_id", type=int, help="docuement ID")
    tf.add_argument("term", type=str, help="term frequency")
    idf = subparsers.add_parser("idf", help="Build inverted document frequencies")
    idf.add_argument("term", type=str, help="term")
    tfidf = subparsers.add_parser("tfidf", help="Build TF-IDF")
    tfidf.add_argument("doc_id", type=int, help="term")
    tfidf.add_argument("term", type=str, help="term")


    args = parser.parse_args()
    
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            
            for i , result in enumerate(results,1):
                print(f"{i}.{result}")
        
        case "build":
            build_command()
        
        case "tf":
            tf = tf_command(args.doc_id , args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id , args.term)
                        
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )




        case _ :
            parser.print_help()



if __name__ == "__main__":
    main()