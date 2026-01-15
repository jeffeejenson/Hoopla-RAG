import argparse
import json
from lib.evaluation import precision_k,precision_k1



def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    results = precision_k(limit)
    for result in results:
        print(f"-Query: {result["query"]}")
        print(f" - Precision@{limit}: {result["precision"]:.4f}")
        print(f" - Recall@{limit}: {result["recall"]:.4f}")
        print(f" - F1 Score: {result["f1"]:.4f}")
        print(f" - Retrieved: {result["retrieved_docs"]}")
        print(f" - Relevant: {result["relevant_docs"]}")


if __name__ == "__main__":
    main()

