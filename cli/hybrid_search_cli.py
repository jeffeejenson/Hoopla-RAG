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



        case _:
            parser.print_help()


if __name__ == "__main__":
    main()