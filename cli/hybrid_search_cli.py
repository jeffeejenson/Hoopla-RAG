import argparse
from hybrid_search import normalise_scores,HybridSearch
from search_util import load_movies


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

            results = hyb.rrf_search(args.query, args.k, args.limit)

            for i,result in enumerate(results,1):
                print(f"{i}. {result["title"]} ")
                print(f"RRF Score :{result["rrf_score"]:.4f}")
                print(f"BM25 RANK : {result["bm25_rank"]:.4f}   ,Semantic Rank : {result["sem_rank"]:.4f}")
                print(f"{result["description"][:100]} ")



        case _:
            parser.print_help()


if __name__ == "__main__":
    main()