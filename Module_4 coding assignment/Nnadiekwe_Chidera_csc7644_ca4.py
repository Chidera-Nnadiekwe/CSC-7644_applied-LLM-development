"""
CSC 7644 - LLM Application Development
Module 4 Coding Assignment: Retrieval-Augmented Generation (RAG) Pipeline

This module implements a complete RAG pipeline with:
- Character-level text chunking with overlap
- BM25 lexical search (baseline retrieval)
- Vector search with OpenAI embeddings + ChromaDB
- Hybrid fusion combining BM25 and vector scores
- Grounded answer generation via OpenAI chat

Why RAG Matters:
- LLMs cannot be retrained on every new fact
- RAG enables dynamic access to external knowledge at inference time
- Reduces hallucination by grounding responses in retrieved evidence
- Essential for building reliable, up-to-date AI applications

INSTRUCTIONS:
- Fix all lines marked with # BUG:
- Complete all sections marked with # STUDENT_COMPLETE
- Do not modify function signatures
- Use only: chromadb, rank-bm25, openai, python-dotenv
- Store API keys in a .env file (never hardcode them)

Required Libraries:
    pip install chromadb rank-bm25 openai python-dotenv

Environment Variables (.env file):
    OPENAI_API_KEY=your_openai_api_key

Author: [Chidera Nnadiekwe]
Date: [April 15, 2026]
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# PROVIDER CLIENT CONFIGURATION

def get_openai_client() -> OpenAI:
    """
    Create and return an OpenAI client using API key from environment.
    
    Returns:
        Configured OpenAI client instance.
        
    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


# TEXT CHUNKING

def chunk_text(text: str, chunk_size: int, stride: int) -> List[str]:
    """
    Split text into overlapping chunks using character-level windowing.
    
    This is a simple but effective chunking strategy that ensures no information
    is lost at chunk boundaries by using overlapping windows.
    
    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        stride: Number of characters to advance between chunks.
                A stride < chunk_size creates overlap.
    
    Returns:
        List of text chunks.
    
    Example:
        >>> chunks = chunk_text("Hello world, this is a test.", 10, 5)
        >>> # Creates overlapping windows of 10 chars, advancing by 5
    """

    chunks = []

    # Edge case handling
    if not text or chunk_size <= 0 or stride <= 0:
        return chunks
    
    start = 0
    text_length = len(text)

    # Slide a fixed-size window across the text with the specified stride
    while start < text_length:
        # Extract a chunk of the specified size
        chunk = text[start: start + chunk_size]

        # Only keep it if there is actual content (non-empty after stripping)
        stripped_chunk = chunk.strip()
        if stripped_chunk:
            chunks.append(stripped_chunk)
        
        # Move the start position forward by the stride
        start += stride

    return chunks


def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """
    Load all .txt files from a directory.
    
    Args:
        data_dir: Path to directory containing text files.
        
    Returns:
        List of dicts with 'filename' and 'content' keys.
    """
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for txt_file in data_path.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            documents.append({
                'filename': txt_file.name,
                'content': content
            })
    
    if not documents:
        print(f"Warning: No .txt files found in {data_dir}")
    
    return documents


# EMBEDDING FUNCTIONS

def get_embeddings(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's embedding API.
    
    Args:
        client: OpenAI client instance.
        texts: List of text strings to embed.
        model: Embedding model name (e.g., 'text-embedding-3-small').
        
    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []
    
    # OpenAI embedding API call
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    
    # Extract embeddings in order
    embeddings = [item.embedding for item in response.data]
    
    return embeddings


# CHROMADB OPERATIONS

def get_chroma_client(db_path: str) -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client.
    
    Args:
        db_path: Path to the database directory.
        
    Returns:
        ChromaDB PersistentClient instance.
    """
    return chromadb.PersistentClient(path=db_path)


def get_or_create_collection(
    chroma_client: chromadb.PersistentClient,
    collection_name: str
) -> chromadb.Collection:
    """
    Get an existing collection or create a new one.
    
    Args:
        chroma_client: ChromaDB client instance.
        collection_name: Name of the collection.
        
    Returns:
        ChromaDB Collection instance.
    """
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )


def upsert_chunks(
    collection: chromadb.Collection,
    chunks: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    ids: List[str]
) -> None:
    """
    Upsert chunks with their embeddings into ChromaDB.
    
    Upsert is idempotent: if an ID already exists, it will be updated.
    
    Args:
        collection: ChromaDB collection.
        chunks: List of text chunks.
        embeddings: List of embedding vectors.
        metadatas: List of metadata dicts for each chunk.
        ids: List of unique IDs for each chunk.
    """
    # BUG Fix: The original condition was missing the ids list and had incorrect logic
    # All lists must be the same length to ensure proper upsert
    if not (len(chunks) == len(embeddings) == len(metadatas) == len(ids)):
        raise ValueError(
            f"Length mismatch: chunks={len(chunks)}, embeddings={len(embeddings)}, "
            f"metadatas={len(metadatas)}, ids={len(ids)}"
        )
    
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


# BM25 SEARCH

def tokenize_for_bm25(text: str) -> List[str]:
    """
    Simple tokenization for BM25: lowercase and split on whitespace.
    
    Args:
        text: Input text to tokenize.
        
    Returns:
        List of lowercase tokens.
    """

    # Convert to lowercase and split on whitespace for consistent tokenization
    return text.lower().split()


def build_bm25_index(documents: List[str]) -> BM25Okapi:
    """
    Build a BM25 index from a list of documents.
    
    Args:
        documents: List of document strings.
        
    Returns:
        BM25Okapi index object.
    """
    tokenized_docs = [tokenize_for_bm25(doc) for doc in documents]
    return BM25Okapi(tokenized_docs)


def bm25_search(
    bm25_index: BM25Okapi,
    query: str,
    documents: List[str],
    top_k: int
) -> List[Tuple[int, float, str]]:
    """
    Search using BM25 and return top-k results.
    
    Args:
        bm25_index: Pre-built BM25 index.
        query: Search query string.
        documents: Original document list (for returning text).
        top_k: Number of results to return.
        
    Returns:
        List of tuples: (doc_index, bm25_score, document_text)
    """
    
    # Tokenize the query using the same tokenization as the documents
    tokenized_query = tokenize_for_bm25(query)
    
    # Score all documents against the query using BM25
    scores = bm25_index.get_scores(tokenized_query)
    
    # Pair each document index with its score and text
    indexed_scores = [(idx, score) for idx, score in enumerate(scores)]
    #
    # Sort the indexed scores in descending order of score
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    # Build the results list with document index, score, and text
    results = []
    for idx, score in indexed_scores[:top_k]:
        results.append((idx, score, documents[idx]))
    
    # Return the results list
    return results


# VECTOR SEARCH

def vector_search(
    collection: chromadb.Collection,
    query_embedding: List[float],
    top_k: int
) -> List[Tuple[str, float, str, Dict]]:
    """
    Search ChromaDB collection using vector similarity.
    
    Note: ChromaDB returns distances, not similarities. For cosine distance,
    similarity = 1 - distance.
    
    Args:
        collection: ChromaDB collection to search.
        query_embedding: Query vector.
        top_k: Number of results to return.
        
    Returns:
        List of tuples: (id, similarity_score, document_text, metadata)
    """
    # BUG Fix: Added "distances" and "metadatas" to the include list to retrieve necessary information.
    # Without distances, we cannot compute similarity scores, and without metadatas, we lose important context.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]  # BUG Fixed: added "distances" and "metadatas"
    )
    
    # Extract results from nested structure
    ids = results['ids'][0] if results['ids'] else []
    documents = results['documents'][0] if results['documents'] else []
    distances = results['distances'][0] if results['distances'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []

    output = []
    for i in range(len(ids)):    
        # BUG Fix: Convert cosine distance to similarity by subtracting from 1.0
        # ChromaDB cosine distance ranges from 0 (identical) to 2 (opposite)
        # So similarity = 1 - distance gives us a range of [1, -1].
        similarity = 1.0 - distances[i]  # Convert distance to similarity
        output.append((
            ids[i], 
            similarity, 
            documents[i], 
            metadatas[i] if i < len(metadatas) else {}
        )) 
    
    return output


# HYBRID FUSION

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max normalize scores to [0, 1] range.
    
    Args:
        scores: List of raw scores.
        
    Returns:
        List of normalized scores.
    """

    # Handle edge case of empty scores list
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)

    # Handle case where all scores are the same
    if min_score == max_score:
        return [1.0] * len(scores)

    # Apply min-max normalization
    return [(s - min_score) / (max_score - min_score) for s in scores]


def hybrid_fusion(
    bm25_results: List[Tuple[int, float, str]],
    vector_results: List[Tuple[str, float, str, Dict]],
    alpha: float = 0.5
) -> List[Tuple[str, float, str]]:
    """
    Combine BM25 and vector search results using weighted score fusion.
    
    Args:
        bm25_results: Results from BM25 search (index, score, text).
        vector_results: Results from vector search (id, similarity, text, metadata).
        alpha: Weight for vector scores (1-alpha for BM25). Default 0.5.
        
    Returns:
        Fused results sorted by combined score: (id/index, fused_score, text)
    """
    
    # Normalize BM25 scores
    bm25_scores = [r[1] for r in bm25_results]
    norm_bm25_scores = normalize_scores(bm25_scores)

    # Normalize vector scores
    vector_scores = [r[1] for r in vector_results]
    norm_vector_scores = normalize_scores(vector_scores)

    # Build a fused dictionary keyed by the document text (or ID) to combine scores
    fused: Dict[str, Dict] = {}

    # Add BM25 results to the fused dictionary
    for (idx, raw_score, text), norm_score in zip(bm25_results, norm_bm25_scores):
        fused[text] = {
            'id': str(idx),  # Use index as ID for BM25 results
            'text': text,
            'bm25_score': norm_score,
            'vector_score': 0.0,  # Default to 0 if not found in vector results
        }

    # Add vector results to the fused dictionary
    for (doc_id, raw_score, text, metadata), norm_score in zip(vector_results, norm_vector_scores):
        key = text  # Use the full text as key to match with BM25 results
        if key in fused:
            # If the text is already in fused (from BM25), update the vector score
            fused[key]['vector_score'] = norm_score
            fused[key]['id'] = doc_id  # Update ID to vector result ID
        else:
            # If not in fused, add it as a new entry
            fused[text] = {
                'id': doc_id,
                'text': text,
                'bm25_score': 0.0,  # Default to 0 if not found in BM25 results
                'vector_score': norm_score,
            }
    
    # Compute fused score and build final results list
    results = []
    for entry in fused.values():
        fused_score = alpha * entry['vector_score'] + (1 - alpha) * entry['bm25_score']
        results.append((entry['id'], fused_score, entry['text']))

    # Sort results by fused score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def format_context(chunks: List[str]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Args:
        chunks: List of retrieved text chunks.
        
    Returns:
        Formatted context string with numbered passages.
    """

    # Handle the case where there are no chunks to format
    if not chunks:
        return "No relevant passages found."

    # Format each chunk as "[Passage N]\n{chunk_text}"
    formatted_chunks = [f"[Passage {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)]

    # Join all formatted passages with "\n\n"
    return "\n\n".join(formatted_chunks)


def generate_grounded_answer(
    client: OpenAI,
    query: str,
    context: str,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate an answer grounded in the retrieved context.
    
    Args:
        client: OpenAI client instance.
        query: User's question.
        context: Retrieved passages formatted as context.
        model: Chat model to use.
        
    Returns:
        Generated answer string.
    """
    system_prompt = """You are a helpful assistant that answers questions based on the provided context passages.

Instructions:
- Only use information from the provided passages to answer the question
- If the answer cannot be found in the passages, say so clearly
- Cite which passage(s) support your answer when possible
- Be concise but complete"""

    user_prompt = f"""Context:
{context}

Question: {query}

Please provide a grounded answer based only on the context above."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )
    
    return response.choices[0].message.content


# INGEST MODE

def run_ingest(
    data_dir: str,
    db_path: str,
    collection_name: str,
    embed_model: str,
    chunk_size: int,
    stride: int
) -> None:
    """
    Ingest documents: chunk, embed, and upsert to ChromaDB.
    
    Args:
        data_dir: Directory containing .txt files.
        db_path: Path for ChromaDB persistence.
        collection_name: Name of the collection.
        embed_model: OpenAI embedding model name.
        chunk_size: Characters per chunk.
        stride: Characters to advance between chunks.
    """
    print(f"Loading documents from {data_dir}...")
    documents = load_documents(data_dir)
    print(f"Found {len(documents)} document(s)")
    
    # Initialize clients
    openai_client = get_openai_client()
    chroma_client = get_chroma_client(db_path)
    collection = get_or_create_collection(chroma_client, collection_name)
    
    total_chunks = 0
    
    for doc in documents:
        filename = doc['filename']
        content = doc['content']
        
        print(f"\nProcessing: {filename}")
        
        # Chunk the document
        chunks = chunk_text(content, chunk_size, stride)
        print(f"  Created {len(chunks)} chunks")
        
        if not chunks:
            continue
        
        # Generate embeddings (batch for efficiency)
        print(f"  Generating embeddings...")
        embeddings = get_embeddings(openai_client, chunks, embed_model)
        
        # Create IDs and metadata
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        
        # Upsert to ChromaDB
        print(f"  Upserting to ChromaDB...")
        upsert_chunks(collection, chunks, embeddings, metadatas, ids)
        
        total_chunks += len(chunks)
    
    print(f"\nIngestion complete! Total chunks: {total_chunks}")
    print(f"Collection '{collection_name}' now has {collection.count()} documents")


# SEARCH MODE

def run_search(
    query: str,
    retriever: str,
    top_k: int,
    db_path: str,
    collection_name: str,
    embed_model: str,
    alpha: float = 0.5
) -> List[Tuple]:
    """
    Search the knowledge base using specified retriever.
    
    Args:
        query: Search query string.
        retriever: One of 'bm25', 'vec', or 'hybrid'.
        top_k: Number of results to return.
        db_path: Path to ChromaDB.
        collection_name: Name of the collection.
        embed_model: Embedding model for vector search.
        alpha: Weight for hybrid fusion (vector weight).
        
    Returns:
        List of search results.
    """
    # Initialize ChromaDB
    chroma_client = get_chroma_client(db_path)
    collection = get_or_create_collection(chroma_client, collection_name)
    
    # Get all documents for BM25 (needed for bm25 and hybrid)
    all_docs = collection.get(include=["documents", "metadatas"])
    documents = all_docs['documents'] if all_docs['documents'] else []
    doc_ids = all_docs['ids'] if all_docs['ids'] else []
    
    if not documents:
        print("No documents found in collection")
        return []
    
    results = []
    
    if retriever == "bm25":
        # BM25 search only
        print(f"Running BM25 search for: '{query}'")
        bm25_index = build_bm25_index(documents)
        results = bm25_search(bm25_index, query, documents, top_k)
        
        # Format results for display
        print(f"\nTop {top_k} BM25 Results:")
        for i, (idx, score, text) in enumerate(results, 1):
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    {text[:200]}...")
    
    # "vec" matches the CLI choices defined in argparse
    elif retriever == "vec":
        # Vector search only
        print(f"Running vector search for: '{query}'")
        openai_client = get_openai_client()
        query_embedding = get_embeddings(openai_client, [query], embed_model)[0]
        results = vector_search(collection, query_embedding, top_k)
        
        # Format results for display
        print(f"\nTop {top_k} Vector Results:")
        for i, (doc_id, similarity, text, metadata) in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {similarity:.4f} | ID: {doc_id}")
            print(f"    {text[:200]}...")
    
    elif retriever == "hybrid":
        # Hybrid fusion of BM25 and vector search
        print(f"Running hybrid search for: '{query}'")
        
        # Get BM25 results
        bm25_index = build_bm25_index(documents)
        bm25_results = bm25_search(bm25_index, query, documents, top_k * 2)
        
        # Get vector results
        openai_client = get_openai_client()
        query_embedding = get_embeddings(openai_client, [query], embed_model)[0]
        vector_results = vector_search(collection, query_embedding, top_k * 2)
        
        # Fuse results
        results = hybrid_fusion(bm25_results, vector_results, alpha)[:top_k]
        
        # Format results for display
        print(f"\nTop {top_k} Hybrid Results (alpha={alpha}):")
        for i, (doc_id, score, text) in enumerate(results, 1):
            print(f"\n[{i}] Fused Score: {score:.4f}")
            print(f"    {text[:200]}...")
    
    else:
        raise ValueError(f"Unknown retriever: {retriever}. Use 'bm25', 'vec', or 'hybrid'")
    
    return results


# ANSWER MODE

def run_answer(
    query: str,
    retriever: str,
    top_k: int,
    db_path: str,
    collection_name: str,
    embed_model: str,
    chat_model: str = "gpt-4o-mini",
    alpha: float = 0.5
) -> str:
    """
    Retrieve relevant chunks and generate a grounded answer.
    
    Args:
        query: User's question.
        retriever: Retrieval method ('bm25', 'vec', or 'hybrid').
        top_k: Number of chunks to retrieve.
        db_path: Path to ChromaDB.
        collection_name: Collection name.
        embed_model: Embedding model name.
        chat_model: Chat model for answer generation.
        alpha: Hybrid fusion weight.
        
    Returns:
        Generated answer string.
    """
    # Retrieve relevant chunks
    print(f"Retrieving top {top_k} chunks using {retriever}...")
    results = run_search(
        query, retriever, top_k, db_path, collection_name, embed_model, alpha
    )
    
    # Extract text from results
    if retriever == "bm25":
        chunks = [r[2] for r in results]  # (idx, score, text)
    elif retriever == "vec":
        chunks = [r[2] for r in results]  # (id, similarity, text, metadata)
    else:  # hybrid
        chunks = [r[2] for r in results]  # (id, score, text)
    
    # Format context
    context = format_context(chunks)
    
    # Generate answer
    print("\nGenerating grounded answer...")
    openai_client = get_openai_client()
    answer = generate_grounded_answer(openai_client, query, context, chat_model)
    
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(answer)
    
    return answer


# MAIN ENTRY POINT

def main():
    """
    Main entry point for the RAG pipeline.
    Parses command line arguments and executes the appropriate mode.
    """
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - CSC 7644 Module 4 Assignment"
    )
    
    # Mode selection
    parser.add_argument(
        'mode',
        type=str,
        choices=['ingest', 'search', 'answer'],
        help="Mode to run: ingest, search, or answer"
    )
    
    # Data and database paths
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/corpus',
        help="Directory containing .txt files for ingestion"
    )
    
    parser.add_argument(
        '--db_path',
        type=str,
        default='./kb',
        help="Path for ChromaDB persistence"
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help="ChromaDB collection name"
    )
    
    # Embedding model
    parser.add_argument(
        '--embed_model',
        type=str,
        default='text-embedding-3-small',
        help="OpenAI embedding model name"
    )
    
    # Chunking parameters
    parser.add_argument(
        '--size',
        type=int,
        default=400,
        help="Chunk size in characters"
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=120,
        help="Stride between chunks (overlap = size - stride)"
    )
    
    # Search parameters
    parser.add_argument(
        '--query',
        type=str,
        help="Search query (required for search and answer modes)"
    )
    
    parser.add_argument(
        '--retriever',
        type=str,
        choices=['bm25', 'vec', 'hybrid'],
        default='hybrid',
        help="Retrieval method"
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help="Number of results to retrieve"
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help="Hybrid fusion weight for vector scores (0-1)"
    )
    
    # Chat model for answer generation
    parser.add_argument(
        '--chat_model',
        type=str,
        default='gpt-4o-mini',
        help="Chat model for answer generation"
    )
    
    args = parser.parse_args()
    
    # Execute appropriate mode
    if args.mode == 'ingest':
        run_ingest(
            data_dir=args.data_dir,
            db_path=args.db_path,
            collection_name=args.collection,
            embed_model=args.embed_model,
            chunk_size=args.size,
            stride=args.stride
        )
    
    elif args.mode == 'search':
        if not args.query:
            parser.error("--query is required for search mode")
        
        run_search(
            query=args.query,
            retriever=args.retriever,
            top_k=args.top_k,
            db_path=args.db_path,
            collection_name=args.collection,
            embed_model=args.embed_model,
            alpha=args.alpha
        )
    
    elif args.mode == 'answer':
        if not args.query:
            parser.error("--query is required for answer mode")
        
        run_answer(
            query=args.query,
            retriever=args.retriever,
            top_k=args.top_k,
            db_path=args.db_path,
            collection_name=args.collection,
            embed_model=args.embed_model,
            chat_model=args.chat_model,
            alpha=args.alpha
        )


if __name__ == "__main__":
    main()