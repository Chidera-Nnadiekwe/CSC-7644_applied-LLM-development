"""
CSC 7644 - LLM Application Development
Module 5 Coding Assignment: Agentic Tool Calling

This module implements a simple agentic pipeline using OpenAI's function calling:
- Three tools with JSON Schema definitions
- Argument validation before execution
- Two-turn controller loop (tool call -> result -> final answer)

Why Agents Matter:
- LLMs can move beyond Q&A to take actions and complete multi-step tasks
- Tool schemas provide type safety and validation
- Disciplined contracts and control separate demos from production systems

INSTRUCTIONS:
- Complete all sections marked with # STUDENT_TODO
- Do not modify the controller flow or remove validation
- Use only: openai, python-dotenv, jsonschema
- Store API keys in a .env file (never hardcode them)

Required Libraries:
    pip install openai python-dotenv jsonschema

Environment Variables (.env file):
    OPENAI_API_KEY=your_openai_api_key

Author: [Chidera Nnadiekwe]
Date: [April 19, 2026]
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from openai import OpenAI
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

# Load environment variables
load_dotenv()


# OPENAI CLIENT

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


# IN-MEMORY KNOWLEDGE BASE

KB_DATA = {
    "VPN": "To connect to the company VPN, use the GlobalProtect client with your SSO credentials. "
           "Contact IT if you need the gateway address.",
    "training": "New employee training modules are available on the Learning Portal. "
                "Complete compliance training within 30 days of start date.",
    "expense": "Submit expenses through Concur. Receipts required for purchases over $25. "
               "Manager approval needed for amounts exceeding $500.",
    "PTO": "Request PTO through Workday at least 2 weeks in advance. "
           "Unused PTO rolls over up to 40 hours annually.",
    "benefits": "Open enrollment for benefits is in November. "
                "Health, dental, and vision plans are available. Contact HR for details."
}


# TOOL SCHEMAS (JSON Schema format for OpenAI function calling)
# OpenAI's current API no longer accepts dots in function names. 
# All dots are replaced with underscores in the function names to comply with OpenAI's requirements.

TOOL_KB_SEARCH = {
    "type": "function",
    "function": {
        "name": "kb_search",   # fixed dot to underscore required by OpenAI's current API
        "description": "Search the company knowledge base for information on a topic. "
                       "Returns relevant information if found.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search term or topic to look up in the knowledge base"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

TOOL_CLOCK_NOW = {
    "type": "function",
    "function": {
        "name": "clock_now",   # fixed dot to underscore required by OpenAI's current API
        "description": "Get the current date and time in UTC.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }
}

# Math_add tool schema definition
TOOL_MATH_ADD = {
    "type": "function",
    "function": {
        "name": "math_add",            # fixed dot to underscore required by OpenAI's current API
        "description": "Add two integers together and return their sum.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "The first integer operand"
                },
                "b": {
                    "type": "integer",
                    "description": "The second integer operand"
                }
            },
            "required": ["a", "b"],
            "additionalProperties": False
        }
    }
}

# List of all available tools
TOOLS = [TOOL_KB_SEARCH, TOOL_CLOCK_NOW, TOOL_MATH_ADD]


# TOOL EXECUTORS
# Define executor functions for each tool. 
def exec_kb_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a knowledge base search.
    
    Args:
        args: Dictionary with 'query' key.
        
    Returns:
        Dictionary with 'result' containing the KB entry or not found message.
    """
    query = args.get("query", "").lower()
    
    # Search for matching entries (case-insensitive partial match)
    for key, value in KB_DATA.items():
        if query in key.lower() or key.lower() in query:
            return {"result": value, "topic": key}
    
    return {"result": f"No information found for '{args.get('query')}'", "topic": None}


# Define executor for clock_now tool
def exec_clock_now(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the current UTC time.
    
    Args:
        args: Empty dictionary (no parameters needed).
        
    Returns:
        Dictionary with current UTC datetime.
    """
    now = datetime.now(timezone.utc)
    return {
        "utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "iso": now.isoformat()
    }


# Define executor for math_add tool
def exec_math_add(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add two integers together.
    
    Args:
        args: Dictionary with 'a' and 'b' integer keys.
        
    Returns:
        Dictionary with 'sum' containing the result.
    """
    
    # Extract integers a and b from args and compute their sum
    a = args["a"]  
    b = args["b"] 
    return {"sum": a + b}  


# Map tool names to their executors
TOOL_EXECUTORS = {
    "kb_search": exec_kb_search,
    "clock_now": exec_clock_now,
    "math_add": exec_math_add   
}

# Map tool names to their parameter schemas (for validation)
TOOL_SCHEMAS = {
    "kb_search": TOOL_KB_SEARCH["function"]["parameters"],
    "clock_now": TOOL_CLOCK_NOW["function"]["parameters"],
    "math_add": TOOL_MATH_ADD["function"]["parameters"]
}


# VALIDATION
# Define a function to validate tool arguments against the corresponding JSON Schema
def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> None:
    """
    Validate tool arguments against the tool's JSON Schema.
    
    Args:
        tool_name: Name of the tool.
        args: Arguments to validate.
        
    Raises:
        ValidationError: If arguments don't match the schema.
        KeyError: If tool name is not recognized.
    """
    if tool_name not in TOOL_SCHEMAS:
        raise KeyError(f"Unknown tool: {tool_name}")
    
    schema = TOOL_SCHEMAS[tool_name]
    validate(instance=args, schema=schema)

# Define a function to execute a tool after validating its arguments
def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and execute a tool.
    
    Args:
        tool_name: Name of the tool to execute.
        args: Arguments for the tool.
        
    Returns:
        Tool execution result as a dictionary.
        
    Raises:
        ValidationError: If arguments are invalid.
        KeyError: If tool is not recognized.
    """
    # Validate arguments before execution (DO NOT REMOVE)
    validate_tool_args(tool_name, args)
    
    # Get and execute the tool
    if tool_name not in TOOL_EXECUTORS:
        raise KeyError(f"No executor found for tool: {tool_name}")
    
    executor = TOOL_EXECUTORS[tool_name]
    return executor(args)


# AGENT CONTROLLER
# Define the main agent controller function that runs the two-turn loop with the model
def run_agent(client: OpenAI, goal: str, model: str = "gpt-4o-mini") -> str:
    """
    Run a two-turn agent controller to accomplish a goal.
    
    Turn 1: Send goal to model, let it pick a tool and supply arguments.
    Turn 2: Execute tool, send result back, get final answer.
    
    Args:
        client: OpenAI client instance.
        goal: The user's goal or question.
        model: Chat model to use.
        
    Returns:
        The final answer from the model.
    """
    print(f"\n{'='*60}")
    print(f"GOAL: {goal}")
    print(f"{'='*60}")
    
    # System prompt for the agent
    system_prompt = """You are a helpful assistant with access to tools. 
When the user asks you to do something, use the appropriate tool(s) to help them.
Available tools:
- kb_search: Search the company knowledge base
- clock_now: Get current UTC time
- math_add: Add two integers

Always use tools when they can help answer the question. 
After receiving tool results, provide a clear, concise final answer."""

    # Initialize conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal}
    ]
    
    # Turn 1: Let the model pick a tool
    print("\n[Turn 1] Sending goal to model...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    messages.append(assistant_message)
    
    # Check if the model wants to call tools
    if not assistant_message.tool_calls:
        # No tool calls, model answered directly
        print("[Turn 1] Model answered without tools")
        return assistant_message.content or "No response generated"
    
    # Process each tool call
    tool_results = []
    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name
        
        # Parse tool arguments
        try:
            tool_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            print(f"[Error] Could not parse tool arguments: {exc}")
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps({"error": f"Invalid JSON arguments: {exc}"})
            })
            continue

        print(f"\n[Turn 1] Tool call: {tool_name}")
        print(f"         Arguments: {tool_args}")

        # Validate and execute the tool
        try:
            result = execute_tool(tool_name, tool_args)
            print(f"         Result: {result}")
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps(result)
            })
        except ValidationError as e:
            print(f"[Error] Validation failed: {e.message}")
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps({"error": f"Validation failed: {e.message}"})
            })
        except Exception as e:
            print(f"[Error] Tool execution failed: {e}")
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": json.dumps({"error": str(e)})
            })
    
    # Add tool results to messages
    messages.extend(tool_results)
    
    # Turn 2: Send tool results back and get final answer
    print("\n[Turn 2] Sending tool results to model...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )
    
    final_message = response.choices[0].message
    
    # Handle case where model wants more tool calls (simplified: just take content)
    if final_message.tool_calls:
        # For this simple controller, we don't do more than 2 turns
        # In production, you'd loop until done or hit a budget
        print("[Turn 2] Model requested more tools (not supported in this simple controller)")
    
    final_answer = final_message.content or "No final answer generated"
    
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(f"{'='*60}")
    print(final_answer)
    
    return final_answer


# MAIN ENTRY POINT
# Define the main function to parse command-line arguments and run the agent
def main():
    """
    Main entry point for the agentic tool calling program.
    """
    parser = argparse.ArgumentParser(
        description="Agentic Tool Calling - CSC 7644 Module 5 Assignment"
    )
    
    parser.add_argument(
        '--goal',
        type=str,
        required=True,
        help="The goal or question for the agent to accomplish"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    
    args = parser.parse_args()
    
    # Initialize client and run agent
    client = get_openai_client()
    run_agent(client, args.goal, args.model)


if __name__ == "__main__":
    main()
    