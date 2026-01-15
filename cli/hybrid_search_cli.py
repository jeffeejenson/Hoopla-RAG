import argparse
from hybrid_search import normalise_command,weighted_search_command,rrf_search_command
from lib.evaluation import llm_judge_results


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
    rrf_search_parser.add_argument("--rerank-method" , type = str ,choices =["individual", "batch","cross_encoder"], help = "rerank method")
    rrf_search_parser.add_argument("--evaluate" , action="store_true", help = "evaluating the result using LLM")





    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalise_command(args.scores)
            for score in scores:
                print(f"* {score:.4f}")

            
        case "weighted-search":
            results = weighted_search_command(args.query,args.alpha,args.limit)

            for i,result in enumerate(results,1):
                print(f"{i}. {result["title"]} ")
                print(f"Hybrid Score :{result["Hybrid Score"]:.4f}")
                print(f"BM25 : {result["BM25"]:.4f}   ,Semantic : {result["Semantic"]:.4f}")
                print(f"{result["Description"][:100]} ")
        
        case "rrf-search":
            results = []
            if args.evaluate:
                results = rrf_search_command(args.query, args.k, args.limit)
                llm_results = llm_judge_results(args.query , results)
                for i,result in enumerate(results,0):
                    print(f"{i+1}{result["title"]}: {llm_results[i]}/3")
                    
            if args.enhance == None:
                results = rrf_search_command(args.query, args.k, args.limit)
                
            elif args.enhance is not None:
                results = rrf_search_command(args.query, args.k, args.limit , args.enhance)
            
            if args.rerank_method == "individual":
                results = rrf_search_command(args.query, args.k, args.limit * 5 , args.enhance , args.rerank_method)
                print("Reranking top 3 results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k=60):")
                for i,result in enumerate(results,1):
                    print(f"{i}. {result["title"]} ")
                    print(f"Rerank Score:{result["reranked_score"]:.4f}")
                    print(f"RRF Score :{result["rrf_score"]:.4f}")
                    print(f"BM25 RANK : {result["bm25_rank"]:.4f}   ,Semantic Rank : {result["sem_rank"]:.4f}")
                    print(f"{result["description"][:100]} ")
                exit(0)
            
            if args.rerank_method == "batch":
                results = rrf_search_command(args.query, args.k, args.limit * 5 , args.enhance , args.rerank_method)
                
                print("Reranking top 3 results using batch method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k=60):")
                for i,result in enumerate(results,1):
                    print(f"{i}. {result["title"]} ")
                    print(f"Rerank rank:{i}")
                    print(f"RRF Score :{result["rrf_score"]:.4f}")
                    print(f"BM25 RANK : {result["bm25_rank"]:.4f}   ,Semantic Rank : {result["sem_rank"]:.4f}")
                    print(f"{result["description"][:100]} ")
                exit(0)


            if args.rerank_method == "cross_encoder":
                results = rrf_search_command(args.query, args.k, args.limit * 5 , args.enhance ,args.rerank_method)
                print("Reranking top 25 results using cross_encoder method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k=60):")
                for i,result in enumerate(results,1):
                    print(f"{i}. {result["title"]} ")
                    print(f"Cross Encoder Score:{result["Cross Encoder Score"]}")
                    print(f"RRF Score :{result["rrf_score"]:.4f}")
                    print(f"BM25 RANK : {result["bm25_rank"]:.4f}   ,Semantic Rank : {result["sem_rank"]:.4f}")
                    print(f"{result["description"][:100]} ")
                exit(0)
                
            for i,result in enumerate(results,1):
                print(f"{i}. {result["title"]} ")
                print(f"RRF Score :{result["rrf_score"]:.4f}")
                print(f"BM25 RANK : {result["bm25_rank"]:.4f}   ,Semantic Rank : {result["sem_rank"]:.4f}")
                print(f"{result["description"][:100]} ")




        case _:
            parser.print_help()


if __name__ == "__main__":
    main()