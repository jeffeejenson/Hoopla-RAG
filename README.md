# Hoopla-RAG

A Retrieval-Augmented Generation (RAG) powered movie search and recommendation engine that lets users find, explore, and ask questions about films using natural language. Hoopla-RAG combines classic IR (BM25 / inverted index) with neural semantic search (sentence-transformer embeddings), hybrid fusion strategies (weighted combination & Reciprocal Rank Fusion), and optional local LLM-based reranking and generation to produce high-quality results and human-friendly answers.

---

## What this project does (TL;DR)
- Indexes a movie catalog (data/movies.json) and supports fast keyword/BM25 search using an inverted index.
- Produces semantic embeddings (sentence-transformers) for whole-movie and chunked descriptions to enable semantic matching via cosine similarity.
- Implements hybrid search strategies:
  - Weighted fusion of normalized BM25 & semantic scores.
  - Reciprocal Rank Fusion (RRF) combining BM25 and semantic ranks.
- Supports reranking:
  - Fast neural reranking using a cross-encoder.
  - LLM-based reranking and scoring with a local LLM (Ollama / Gemma).
- Provides Retrieval-Augmented Generation (RAG): use search results as evidence and synthesize answers (citations, summaries, question answering) via a local LLM.
- CLI entry points let you: build indices, run BM25 and semantic queries, run hybrid / RRF searches, create chunk embeddings, produce RAG answers, and evaluate results.

---

## Key components & entry points
Top-level CLI scripts:
- cli/keyword_search_cli.py — build inverted index, compute TF/IDF/BM25 components, run BM25 searches.
- cli/semantic_search_cli.py — embed texts, create chunked embeddings, run semantic search.
- cli/hybrid_search_cli.py — run weighted hybrid search and RRF search; supports query enhancement and reranking options.
- cli/augmented_generation_cli.py — run RAG-style queries, summarization, citations, and question answering using search results as context.
- cli/evaluation_cli.py — (evaluation utilities).

Core libs (cli/lib):
- keyword_search.py — inverted_index class, tokenization, TF/IDF, BM25 implementation, and BM25 CLI helpers.
- semantic_search.py — SentenceTransformer-based embeddings, chunked semantic search, chunking helpers.
- hybrid_search.py — orchestrates BM25 + semantic, weighted fusion, RRF, normalization helpers and command wrappers.
- augemented_generation.py — functions that call a local LLM (via ollama) to produce RAG outputs (answers, summaries, citations).
- rerank_local.py & rerank.py — reranking modules (local vs remote/cross-encoder/inference).
- enhance_local.py & enhance.py — query enhancement (spell, rewrite, expand) local vs remote variants.
- evaluation.py — utilities to evaluate results (LLM judge wrappers, etc).

Cache & data:
- Build you're cache using load/save functions (automated through CLI)
- data/movies.json — source movie dataset used to build indices & embeddings.
- data/stopwords.txt — stopwords used in tokenization.
- cache/ — on-disk caches (index.pkl, docmap.pkl, term_frequencies.pkl, doc_lengths.pkl, movie_embeddings.npy, chunk_embeddings.npy, chunk_metadata.json).

Project metadata:
- pyproject.toml — project dependencies (google-genai, nltk, numpy, python-dotenv, sentence-transformers listed).

---

## Search algorithms used (details)

1. Tokenization & preprocessing
- Lowercasing, punctuation removal, splitting by whitespace.
- Stopwords removed using data/stopwords.txt.
- Porter stemming applied (nltk.stem.PorterStemmer).

2. Inverted index & term frequencies
- inverted_index maintains:
  - an index mapping token -> set(docIDs)
  - term_frequencies: Counter for each document
  - doc_lengths for BM25 length normalization
- Persisted as pickle files in cache/.

3. TF, IDF, TF-IDF
- TF: raw term counts per document.
- IDF: computed as log((N + 1) / (df + 1)) in code (safeguarded).
- TF-IDF: product of TF and IDF.

4. BM25 (stateful local BM25 implementation)
- BM25 TF component:
  - tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
- BM25 IDF:
  - idf = log((N - df + 0.5) / (df + 0.5) + 1)
- Tunable parameters:
  - k1 (default 1.5)
  - b  (default 0.75)
- The BM25 search sums BM25 scores of query tokens across docs and ranks by total.

5. Semantic search (dense retrieval)
- Uses sentence-transformers model "all-MiniLM-L6-v2".
- Two modes:
  - Document-level embeddings (title + description) for global semantic search using cosine similarity.
  - Chunked semantic search: each movie description is split to semantic chunks (sentence-based chunking via semantic_chunk1), each chunk encoded, and search aggregates chunk-level relevance to movie-level scores (max across chunks).
- Cosine similarity computed and used for ranking.

6. Hybrid fusion strategies
- Weighted combination:
  - Normalize BM25 and semantic scores to [0,1] using min-max normalization.
  - Combined score = alpha * BM25 + (1 - alpha) * semantic.
  - alpha is tunable when running weighted search.
- Reciprocal Rank Fusion (RRF):
  - Compute ranks for BM25 and semantic lists.
  - RRF score for a document = sum(1 / (k + rank_i)) across the ranking lists (k default 60).
  - RRF is robust and helps combine complementary lists.

7. Reranking (improve top-k ordering)
- Cross-encoder reranking using SentenceTransformers CrossEncoder ("cross-encoder/ms-marco-TinyBERT-L2-v2").
  - Pairs query & candidate text; cross-encoder returns a relevance score for final ordering.
- LLM-based reranking (local LLM with Ollama):
  - Individual scoring: ask the LLM to rate each movie 0–10 and use that as score.
  - Batch ranking: ask the LLM to return an ordered list of IDs (as JSON).

8. Retrieval-Augmented Generation (RAG)
- After retrieving top-k results (RRF/hybrid/BM25/semantic), feed formatted evidence to the LLM to:
  - Answer user questions.
  - Generate summaries or citation-aware answers (prompt includes instructions on citation formatting).
- Local LLM (Ollama/gemma3) used for generation in provided code.

---

## Local LLMs & models used
This repository is designed to run fully locally (supported by local models), using:

- Ollama + Gemma
  - The code uses the ollama Python client (import ollama) and invokes ollama.generate(...).
  - Model referenced: 'gemma3:4b' (used in augemented_generation.py and rerank_local.py).
  - Purpose:
    - LLM-based reranking (individual scores, batch ranking).
    - RAG answer generation (rag, summarize, citations, question commands).
  - Note: You must install Ollama on your machine and pull the gemma model (or other compatible local model) to use these features.

- sentence-transformers
  - Embedding model: "all-MiniLM-L6-v2" — small, CPU-friendly, good semantic performance.
  - Cross-encoder for reranking: "cross-encoder/ms-marco-TinyBERT-L2-v2".

- (Optional) Google GenAI dependency is included in pyproject, but current code primarily uses Ollama for local LLM calls. If you want to swap to a cloud LLM, adapt augemented_generation.py and rerank modules accordingly.

---

## Installation & run (run on any machine)
Minimum supported Python: 3.13 (pyproject specifies requires-python >= 3.13).

1) Clone the repo
- git clone https://github.com/jeffeejenson/Hoopla-RAG.git
- cd Hoopla-RAG

2) Create & activate virtual environment
- Unix/macOS:
  - python3 -m venv .venv
  - source .venv/bin/activate
- Windows:
  - python -m venv .venv
  - .venv\Scripts\activate

3) Upgrade pip and install dependencies
- Recommended (install core dependencies):
  - pip install --upgrade pip
  - pip install numpy python-dotenv nltk sentence-transformers
  - pip install ollama           # if available via pip for the Python client; otherwise follow Ollama client install instructions
  - pip install torch           # sentence-transformers often needs torch (CPU/GPU)
- Or install individual packages from pyproject:
  - pip install google-genai==1.57.0 nltk==3.9.1 numpy python-dotenv sentence-transformers

4) (Optional) If you will use LLM-based features:
- Install and set up Ollama (OS-specific instructions on the Ollama site).
- Pull a model (example): ollama pull gemma3:4b
- Verify Ollama CLI and that the model name matches code references: 'gemma3:4b'.

5) Prepare NLTK (if needed)
- The repository uses PorterStemmer only (no NLTK corpora required). If you add NLTK functionality, run:
  - python -c "import nltk; nltk.download('punkt')"

6) Data
- Ensure data/movies.json is present (it is included in repo). Stopwords in data/stopwords.txt are included.

7) Create the cache directory (if not present)
- mkdir cache

---

## Preparing caches & indices (first-time run)
1) Build inverted index (BM25 / keyword)
- python cli/keyword_search_cli.py build
- This will create files under cache/: index.pkl, docmap.pkl, term_frequencies.pkl, doc_lengths.pkl

2) Create document embeddings (semantic search)
- python cli/semantic_search_cli.py embed_text "test"  # quick verification
- To generate and cache whole-document embeddings:
  - Use commands in semantic_search_cli.py or call via Python functions:
  - python - <<PY
    from cli.lib.semantic_search import SemanticSearch
    from cli.search_util import load_movies
    s = SemanticSearch()
    s.load_or_create_embeddings(load_movies())
    PY
  - Or use the CLI to create chunked embeddings:
    - python cli/semantic_search_cli.py embed_chunks

After building, cached arrays are stored under cache/ (movie_embeddings.npy or cache/chunk_embeddings.npy + chunk_metadata.json).

---

## How to run: CLI examples & entry points

1) Keyword / BM25 search
- Build index (first time):
  - python cli/keyword_search_cli.py build
- Search for a simple keyword query:
  - python cli/keyword_search_cli.py search "family bear movie"
  - Example output: list of movie titles matching query tokens
- BM25 full scoring:
  - python cli/keyword_search_cli.py bm25search "family movie about bears"

Debugging TF / IDF / BM25 internals:
- python cli/keyword_search_cli.py tf 10 "bear"
- python cli/keyword_search_cli.py idf "bear"
- python cli/keyword_search_cli.py bm25tf 10 "bear" 1.5 0.75

2) Semantic search
- Verify model & embedding:
  - python cli/semantic_search_cli.py verify
  - python cli/semantic_search_cli.py embed_text "A sample query"
- Run semantic search:
  - python cli/semantic_search_cli.py search "family movie about bears"
- Create chunk embeddings:
  - python cli/semantic_search_cli.py embed_chunks
- Search chunked semantic indices:
  - python cli/semantic_search_cli.py search_chunked "movie about time travel" --limit 5

3) Hybrid search
- Weighted fusion:
  - python cli/hybrid_search_cli.py weighted-search "crime thriller with twist" --alpha 0.6 --limit 5
- RRF fusion:
  - python cli/hybrid_search_cli.py rrf-search "romantic comedies with witty banter" -k 60 --limit 5
- RRF with query enhancement and reranking methods:
  - python cli/hybrid_search_cli.py rrf-search "family movie with animals" --enhance spell --rerank-method cross_encoder --limit 10
  - Rerank methods available: individual (LLM scoring), batch (LLM returns id order), cross_encoder (neural reranker)

4) RAG / Augmented generation
- RAG (search + generate answer):
  - python cli/augmented_generation_cli.py rag "Recommend movies like The Iron Giant"
  - Output: printed search results and a LLM-generated answer based on retrieved documents.
- Summarize:
  - python cli/augmented_generation_cli.py summarize "sci-fi movies with ethical dilemmas" --limit 5
- Citations:
  - python cli/augmented_generation_cli.py citations "best movies about journalism" --limit 5
- Direct question answering:
  - python cli/augmented_generation_cli.py question "Which movie in the dataset features a detective in New York?" --limit 5

5) Example queries (copy-paste)
- "family movie about bears in the woods"
- "crime thriller with an unexpected twist ending"
- "romantic comedies from the 1990s"
- "movies where the protagonist travels back in time"
- "recommend movies like The Iron Giant"
- "best documentaries about climate change"

---

## Expected cache & artifact files
- cache/index.pkl
- cache/docmap.pkl
- cache/term_frequencies.pkl
- cache/doc_lengths.pkl
- cache/movie_embeddings.npy (whole-document embeddings)
- cache/chunk_embeddings.npy
- cache/chunk_metadata.json

---

## Troubleshooting & tips
- If semantic models complain about missing torch or slow CPU runs:
  - Install torch appropriate to your system (CPU/GPU) before installing sentence-transformers:
    - pip install torch --index-url https://download.pytorch.org/whl/cpu  (or use official instructions)
- If Ollama features fail:
  - Install Ollama from https://ollama.com and ensure the model name (gemma3:4b) is available locally (`ollama ls`).
  - Match the model name in code or change to a model you have.
- Permission or file path errors:
  - Ensure cache/ and data/ directories are readable/writable.
- Rebuild index or embeddings when data/movies.json changes.
- If stopwords removal appears too aggressive: edit data/stopwords.txt.

---

## Code structure quick reference
- cli/
  - keyword_search_cli.py
  - semantic_search_cli.py
  - hybrid_search_cli.py
  - augmented_generation_cli.py
  - evaluation_cli.py
  - lib/
    - keyword_search.py (inverted index & BM25)
    - semantic_search.py (embeddings & chunked semantic)
    - hybrid_search.py (fusion & top-level hybrid calls)
    - augemented_generation.py (RAG prompt wrappers)
    - rerank_local.py / rerank.py (local & remote rerankers)
    - enhance_local.py / enhance.py (query enhancement)
    - evaluation.py (LLM judge calls)
- data/
  - movies.json
  - stopwords.txt
- cache/ (created at runtime)

---

## Notes on reproducibility & deployment
- This project is designed to run locally for prototyping and experimentation.
- For production-scale deployments:
  - Use a persistent vector database (FAISS, Milvus, Pinecone, etc.) instead of the in-repo numpy cache.
  - Use a scalable LLM endpoint or a containerized local LLM serving for parallelism.
  - Add rate-limits, batching, and async flows for LLM calls.
  - Add unit tests and CI to validate retrieval quality.

---

 
