import argparse
from lib.augemented_generation import rag_command,summarise_command,citations_command,question_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarise_parser = subparsers.add_parser(
        "summarize" , help = "summarise the content"
    )
    summarise_parser.add_argument("query", type=str, help="Search query for RAG" )
    summarise_parser.add_argument("--limit", type=int, default=5,help="limit" )

    citation_parser = subparsers.add_parser(
        "citations" , help = "provide citation"
    )
    citation_parser.add_argument("query", type=str, help="Search query for RAG" )
    citation_parser.add_argument("--limit", type=int, default=5,help="limit" )

    question_parser = subparsers.add_parser(
        "question" , help = "provide answer to question"
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG" )
    question_parser.add_argument("--limit", type=int, default=5,help="limit" )






    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_result = rag_command(query)
            rrf_results = rag_result["results"]
            rag_response = rag_result["rag-response"]
            print("Search Results:")
            for i , rrf_result in enumerate(rrf_results):
                print(f"{i}.{rrf_result["title"]}")
            print("RAG Results:")
            print(rag_response)
        
        case "summarize":
            summarise_result = summarise_command(args.query,args.limit)
            rrf_results = summarise_result["results"]
            summarise_respone = summarise_result["summarise-respone"]
            print("Search Results:")
            for i , rrf_result in enumerate(rrf_results):
                print(f"{i}.{rrf_result["title"]}")
            print("RAG Results:")
            print(summarise_respone)
        
        case "citations":
            citations_result = citations_command(args.query,args.limit)
            rrf_results = citations_result["results"]
            citation_respone = citations_result["citation-respone"]
            print("Search Results:")
            for i , rrf_result in enumerate(rrf_results):
                print(f"{i}.{rrf_result["title"]}")
            print("RAG Results:")
            print(citation_respone)

        case "question":
            question_result = question_command(args.query,args.limit)
            rrf_results = question_result["results"]
            question_response = question_result["question-respone"]
            print("Search Results:")
            for i , rrf_result in enumerate(rrf_results):
                print(f"{i}.{rrf_result["title"]}")
            print("Answers:")
            print(question_response)







            

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()