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
            try:
                idx.load() 
                print(idx.get_tf(args.doc_id, args.term))
            except Exception as e:
                print(f"Error loading index: {e}")
        case "idf":
            try:
                idx.load()
                tokens = tokenize_text(args.term)
                if len(tokens) != 1:
                     raise ValueError(f"IDF expects a single word, got: {tokens}")
                
                target_token = tokens[0]
                total_doc_count = len(idx.docmap)
                term_match_docs = idx.get_documents(target_token)
                term_match_doc_count = len(term_match_docs)
                i = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
                print(f"Inverse document frequency of '{args.term}': {i:.2f}")
            except Exception as e:
                print(f"Error loading index: {e}")
        
        case "tfidf":
            try:
                idx.load() 
                tf = idx.get_tf(args.doc_id, args.term)
            except Exception as e:
                print(f"Error loading index: {e}")
            try:
                idx.load()
                tokens = tokenize_text(args.term)
                if len(tokens) != 1:
                     raise ValueError(f"IDF expects a single word, got: {tokens}")
                
                target_token = tokens[0]
                total_doc_count = len(idx.docmap)
                term_match_docs = idx.get_documents(target_token)
                term_match_doc_count = len(term_match_docs)
                i = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            except Exception as e:
                print(f"Error loading index: {e}")
            tf_idf = tf * i
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")





        case _ :
            parser.print_help()



if __name__ == "__main__":
    main()