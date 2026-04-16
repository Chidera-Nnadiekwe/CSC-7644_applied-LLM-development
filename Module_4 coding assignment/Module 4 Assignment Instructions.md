# Module 4 Assignment Instructions
Most real applications don’t run frontier models locally; instead, they call them over APIs. The difference between a demo and a dependable system often comes down to output structure, error handling, and smart batching. In this assignment, you’ll learn to (1) form well-scoped chat messages, (2) demand machine-readable responses that downstream code can trust, and (3) use the official Batch API for cheaper, high-throughput, non-interactive workloads. These skills directly affect cost, latency, and reliability in production. 

Real RAG systems succeed or fail on the “glue”: chunking strategy, indexing choices, retrieval quality, score fusion, and disciplined prompting, far more than on model size alone. By debugging and completing this pipeline, you learn how to persist embeddings, control costs, interpret distances vs. similarities, and produce auditable, grounded outputs that downstream code can trust. These skills translate directly to production settings and set you up for future modules. 

## Instructions
You are given a single Python script that implements the core of a RAG pipeline: 

- Character-level chunking
- BM25 lexical index (baseline) 
- Vector search with OpenAI embeddings + ChromaDB 
- Hybrid fusion (BM25 + vector) 
- Simple grounded answer generation via OpenAI chat 

Parts of the code are marked with # STUDENT_COMPLETE, and a few lines are intentionally broken, marked with # BUG. Your job is to fix the bugs and complete the TODOs so the script runs end-to-end on a small local corpus. No rerankers are used in this assignment. 

**Context**: In production, RAG, retrieval quality, and glue code correctness determine whether your system is reliable. This lab helps you become comfortable with the moving parts—chunking, indexing, searching, fusing, and grounding—without requiring you to build a full app. 

**Allowed libraries** 
- chromadb, rank-bm25, openai, python-dotenv
- Python 3.12 
- No other third-party packages. 

*Put API keys in a .env file.* 

**Tasks you’ll perform** 
1. Implement/complete small functions (chunking, normalization, fusion, prompt formatting). 
2. Fix a few intentional bugs (Chroma include list, distance→similarity, ID/length mismatch, retrieval mode branch). 
3. Run three modes: ingest, search, and answer.
**Minimal Corpus you provide** 

Create a minimal corpus with two text files from Project Gutenberg. The first text file is the novel Beowulf, and the second is The Adventures of Sherlock Holmes. Download the raw text versions and store them in .txt. files. 

How to run (after fixing the code) 
1. **Ingest: chunk + embed + upsert** 
    1. python LastName_FirstName_csc7644_ca4.py ingest \ 
        - --data_dir data/corpus \ 
        - --db_path ./kb \ 
        - --collection novel \ 
        - --embed_model text-embedding-3-small \ 
        - --size 400 --stride 120 

2. **Search (choose one retriever)** 
    1. python LastName_FirstName_csc7644_ca4.py search \ 
        - --query "Where was Holmes sitting when he said: Try the settee?" \ 
        - --retriever bm25 --top_k 5 \ 
        - --db_path ./kb --collection novel --embed_model text-embedding-3-small 
    2. python LastName_FirstName_csc7644_ca4.py search \ 
        - --query "How is the road around the corner tfrom he retired Saxe-Coburg Square described?" \ 
        - --retriever vec --top_k 5 \ 
        - --db_path ./kb --collection novel --embed_model text-embedding-3-small  

**Submission Guidelines**
Submit a single Python file named LastName_FirstName_csc7644_ca4.py to the Moodle submission link. Do not rename any required function or mode names. Follow PEP-8, include docstrings for public functions, and add inline comments where logic isn’t obvious.

**Check Before You Submit**
1. File name exactly LastName_FirstName_csc7644_ca4.py 
2. All # STUDENT_COMPLETE areas implemented 
3. All # BUG: lines corrected so modes run end-to-end 
4. Uses .env (no keys in source and do not submit the .env file) 
5. Runs with commands above on the local corpus containing the two novels 
6. Clear inline comments & docstrings 