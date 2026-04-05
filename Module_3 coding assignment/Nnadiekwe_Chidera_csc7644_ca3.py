"""
CSC 7644 - LLM Application Development
Module 3 Coding Assignment: OpenAI API & Batch Processing

This module demonstrates how to interact with LLM APIs in production settings:
- Basic chat completions with OpenAI and OpenRouter
- Structured outputs using JSON Schema for reliable parsing
- Batch processing for high-throughput, cost-effective workloads

Why This Matters:
- Most real applications call frontier models over APIs, not locally
- Output structure, error handling, and smart batching separate demos from production
- These skills directly affect cost, latency, and reliability

INSTRUCTIONS:
- Complete all functions marked with # STUDENT_COMPLETE
- Do not modify function signatures
- Use only built-in modules plus: openai, python-dotenv
- Store API keys in a .env file (never hardcode them)
- Follow PEP-8 style guidelines

Required Libraries:
    pip install openai python-dotenv

Environment Variables (.env file):
    OPENAI_API_KEY=your_openai_api_key
    OPENROUTER_API_KEY=your_openrouter_api_key

Author: [Chidera C. Nnadiekwe]
Date: [April 4, 2026]
"""

import os
import json
import time
import argparse
from typing import List, Dict, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PROVIDER CLIENT CONFIGURATION
def get_client(provider: str) -> OpenAI:
    """
    Create and return an OpenAI SDK client configured for the specified provider.
    
    Args:
        provider: Either 'openai' or 'openrouter'.
        
    Returns:
        A configured OpenAI client instance.
        
    Raises:
        ValueError: If provider is not recognized.
        EnvironmentError: If required API key is not set.
    """

    if provider == 'openai':
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise EnvironmentError('OPENAI_API_KEY not found in environment'
            )
        # Use default base URL for OpenAI
        return OpenAI(api_key=key)
    
    elif provider == 'openrouter':
        key = os.getenv('OPENROUTER_API_KEY')
        if not key:
            raise EnvironmentError('OPENROUTER_API_KEY not found in environment'
            )
        # OpenRouter uses an OpenAI-compatible endpoint
        return OpenAI(
            api_key=key, 
            base_url='https://openrouter.ai/api/v1'
        )
    
    else:
        raise ValueError(f'Unknown provider: {provider}'
        )


# CHAT MODE
def run_chat(client: OpenAI, model: str) -> dict:
    """
    Send a minimal chat completion request and return the response with token usage.
    
    This demonstrates a basic working call with proper role configuration
    and parameter settings for controlled output.
    
    Args:
        client: Configured OpenAI client.
        model: Model identifier (e.g., 'gpt-4o-mini').
        
    Returns:
        Dictionary with keys:
            - content: The assistant's response text
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
            - total_tokens: Total tokens used
    """

    # Define messages with system and user roles
    messages = [
        {"role": "system", "content": "You are a patient, concise assistant."},
        {"role": "user", "content": "In one sentence, explain what a prompt does in an LLM."}
    ]

    # Call the API
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        top_p=1,
        max_tokens=128
    )

    content = resp.choices[0].message.content

    # Extract token usage, defaulting to 0 if not available
    usage = getattr(resp, 'usage', None)
    return {
        'content': content,
        'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
        'completion_tokens': getattr(usage, 'completion_tokens', 0),
        'total_tokens': getattr(usage, 'total_tokens', 0)
    }


# STRUCTURED OUTPUTS (JSON SCHEMA)
def extract_invoice_json(client: OpenAI, model: str, text: str) -> dict:
    """
    Extract structured invoice data from text using JSON Schema enforcement.
    
    This demonstrates structured outputs where the model is constrained to
    return valid JSON matching a specific schema, enabling reliable parsing.
    
    Args:
        client: Configured OpenAI client.
        model: Model identifier (e.g., 'gpt-4o-mini').
        text: Input text containing invoice information.
        
    Returns:
        Dictionary with extracted fields:
            - invoice_number: string or null
            - invoice_date: string (YYYY-MM-DD) or null
            - vendor: string or null
            - total_amount_usd: number or null
    """
    
    # Define JSON Schema for invoice extraction
    json_schema = {
        "name": "invoice_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "invoice_number": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "invoice_date": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "vendor": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "total_amount_usd": {
                    "anyOf": [{"type": "number"}, {"type": "null"}]
                }
            },
            "required": [
                "invoice_number",
                "invoice_date",
                "vendor",
                "total_amount_usd"
            ],
            "additionalProperties": False
        }
    }

    # Define messages with system instructions and user input
    messages = [
        {"role": "system", "content": "Extract only the following fields and return JSON only. Set missing fields to null. Use YYYY-MM-DD format for dates."},
        {"role": "user", "content": text}
    ]

    # Call the API with JSON Schema response format
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_schema", "json_schema": json_schema}
    )

    # Return the extracted invoice data
    content = resp.choices[0].message.content
    return json.loads(content)

# BATCH MODE
def build_batch_manifest(items: List[dict]) -> List[dict]:
    """
    Build a batch manifest from a list of input items.
    
    Each item becomes a request object formatted for the OpenAI Batch API.
    This allows processing multiple requests in a single batch job for
    cost savings and higher throughput.
    
    Args:
        items: List of dictionaries, each containing 'text' to process.
        
    Returns:
        List of request objects, each with:
            - custom_id: Unique identifier (e.g., 'inv-0001')
            - method: HTTP method ('POST')
            - url: API endpoint
            - body: Request body with model, messages, etc.
    """

    manifest = []
    for idx, item in enumerate(items):
        req = {
            "custom_id": f"inv-{idx+1:04d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "Extract invoice info from text and return as JSON object only."},
                    {"role": "user", "content": item.get("text", "")}
                ]
            }
        }
        manifest.append(req)
    return manifest


# Utility function to write JSONL files for batch manifests
def write_jsonl(rows: List[dict], path: str) -> None:
    """
    Write a list of dictionaries to a JSONL file (one JSON object per line).
    
    JSONL (JSON Lines) is the required format for OpenAI Batch API manifests.
    
    Args:
        rows: List of dictionaries to write.
        path: File path to write to.
        
    Returns:
        None
    """

    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n') # Write each dict as a JSON object on its own line

    # Print confirmation of how many items were written and the file path
    print(f"Wrote {len(rows)} items to {path}")


# Utility function to run a batch job and handle the full workflow
def run_batch(client: OpenAI, manifest_path: str) -> str:
    """
    Execute a batch job: upload manifest, create batch, poll until complete, save results.
    
    The Batch API is designed for high-throughput, non-interactive workloads
    with significant cost savings compared to synchronous requests.
    
    Args:
        client: Configured OpenAI client.
        manifest_path: Path to the JSONL manifest file.
        
    Returns:
        Path to the results file ('results.jsonl').
    """
    # Step 1: Upload manifest file
    with open(manifest_path, 'rb') as f:
        uploaded = client.files.create(file=f, purpose='batch')
    file_id = uploaded.id
    print(f"Uploaded manifest file: {file_id}")

    # Step 2: Create the batch job
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    batch_id = batch.id
    print(f"Created batch job: {batch_id}")

    # Step 3: Poll for completion
    while True:
        status = client.batches.retrieve(batch_id)
        if status.status == 'completed':
            break
        elif status.status in ['failed', 'expired', 'cancelled']:
            print(f"Batch job {batch_id} ended with status: {status.status}")
            return ""
        print(f"Batch {batch_id} status: {status.status}, waiting 10s...")
        time.sleep(10)
    
    # Debug
    print(f"Output file ID: {status.output_file_id}")
    print(f"Error file ID: {status.error_file_id}")
    print(f"Request counts: {status.request_counts}")

    # Step 4: Download results
    output_file_id = status.output_file_id or status.error_file_id
    if not output_file_id:
        print("Failed to retrieve results file.")
        return ""
    
    results = client.files.content(output_file_id)
    results_path = "results.jsonl"
    with open("results.jsonl", "wb") as f:
        f.write(results.read())

    # Step 5: Return results path
    return results_path

# SUMMARIZE MODE
def summarize_text(client: OpenAI, model: str, text: str) -> dict:
    """
    Summarize the provided text using the LLM.
    
    Args:
        client: Configured OpenAI client.
        model: Model identifier.
        text: Text to summarize.
        
    Returns:
        Dictionary with summary and token usage.
    """

    # Define messages for summarization task
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides concise summaries."},
        {"role": "user", "content": f"Please summarize the following text in 2-3 sentences:\n\n{text}"}
    ]

    # Call the API
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256
    )

    # Extract summary content and token usage
    content = resp.choices[0].message.content

    # Extract token usage, defaulting to 0 if not available
    usage = getattr(resp, 'usage', None)
    return {
        'content': content,
        'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
        'completion_tokens': getattr(usage, 'completion_tokens', 0),
        'total_tokens': getattr(usage, 'total_tokens', 0)
    }


# MAIN ENTRY POINT
def main():
    """
    Main entry point for the OpenAI API program.
    Parses command line arguments and executes the appropriate mode.
    """
    # Define command line arguments for different modes and options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', 
        choices=['chat', 'struct', 'summarize', 'batch-prepare', 'batch-run']
        )
    parser.add_argument(
        '--provider', 
        choices=['openai', 'openrouter'], default='openai'
        )
    parser.add_argument('--model', default=None)
    parser.add_argument('--text', default=None)
    parser.add_argument('--manifest', default='tasks.jsonl')
    parser.add_argument('--window', default='24h')
    args = parser.parse_args()

    # Set default models based on provider if not specified
    if args.model is None:
        args.model = 'gpt-4o-mini' if args.provider == 'openai' else 'meta-llama/llama-3.1-405b-instruct'

    if args.mode == 'chat':
        client = get_client(args.provider)
        result = run_chat(client, args.model)
        print(json.dumps(result, indent=2))

    elif args.mode == 'struct':
        client = get_client(args.provider)
        text = args.text or "Invoice #44921 from Acme Co. Date: 2025-08-28. Total: $4,912."
        result = extract_invoice_json(client, args.model, text)
        print(json.dumps(result, indent=2))

    elif args.mode == 'summarize':
        if not args.text:
            print("--text is required for summarize mode")
            return
        client = get_client(args.provider)
        result = summarize_text(client, args.model, args.text)
        print(json.dumps(result, indent=2))

    elif args.mode == 'batch-prepare':
        items = [
            {"text": "Invoice #1001 from Alpha Co. Date: 2025-01-01. Total: $500."},
            {"text": "Invoice #1002 from Beta LLC. Date: 2025-02-15. Total: $1,250."},
            {"text": "Invoice #1003 from Gamma Inc. Date: 2025-03-20. Total: $3,000."}
        ]
        manifest = build_batch_manifest(items)
        write_jsonl(manifest, args.manifest)

    elif args.mode == 'batch-run':
        if not os.path.exists(args.manifest):
            print(f"Manifest file {args.manifest} not found.")
            return
        if args.provider != 'openai':
            print("Batch run only supports OpenAI provider.")
            return
        client = get_client('openai')
        results_path = run_batch(client, args.manifest)
        print(f"Batch results saved to {results_path}")

if __name__ == "__main__":
    main()
