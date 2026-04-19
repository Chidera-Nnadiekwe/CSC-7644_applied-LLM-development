# CSC 7644 – Module 5 Coding Assignment: Agentic Tool Calling

## Overview

Agents fail or succeed on disciplined contracts and control: tight tool schemas, validation and repair, simple but enforced budgets, clear stop rules, and auditable traces. By finishing this controller and passing a golden trace, you'll practice the habits production teams rely on: typed I/O, deterministic execution paths, observability, and safe fallbacks.

---

## Instructions

You are given a single Python script that implements the core of a simple agentic pipeline using the OpenAI Chat Completions API with tool/function calling. The script includes three tools and a two-turn controller; several lines are marked with `# STUDENT_TODO` which you must complete.

---

## What's in the Script

**Three tools (JSON Schemas + executors):**

- `kb.search` — searches a tiny in-memory KB *(provided)*
- `clock.now` — returns current time *(provided)*
- `math.add` — **you implement** the JSON Schema and a 3–5 line executor

**Validation:** arguments must pass JSON Schema validation before a tool runs.

**Controller (2 turns):**
- Turn 1 lets the model pick a tool and supply arguments
- Turn 2 feeds the tool's result back and asks for the final answer

---

## Your Tasks

1. Fill in the `math.add` JSON Schema (strict: integers only, required fields, no extra props).
2. Implement `exec_math_add` to return `{"sum": a + b}`.
3. Parse tool arguments from `call.function.arguments` (JSON string → dict).
4. Ensure validation is called and keep the provided controller flow intact.

---

## Allowed Libraries

- `openai`
- `python-dotenv`
- `jsonschema`
- Python 3.12
- No other third-party packages
- Put your API key in a `.env` file (`OPENAI_API_KEY=...`). **Do not hard-code keys.**

---

## How to Run

**Install dependencies:**
```bash
pip install --upgrade openai python-dotenv jsonschema
```

**Set your API key in the `.env` file:**
```
OPENAI_API_KEY=sk-...your key...
```

**Run the agent with a goal:**
```bash
python LastName_FirstName_csc7644_ca5_easy.py --goal "Add 12 and 34, then show current time and check KB for 'VPN'."
```

**Try a few more goals:**
```bash
# Pure math
python LastName_FirstName_csc7644_ca5_easy.py --goal "What is 120 + 45?"

# Time only (UTC)
python LastName_FirstName_csc7644_ca5_easy.py --goal "What time is it now in UTC?"

# KB lookup
python LastName_FirstName_csc7644_ca5_easy.py --goal "Search the KB for 'training' and summarize."
```

If everything is wired correctly, you'll see a concise **FINAL ANSWER** printed to stdout. If you encounter a validation error, verify your `math.add` schema, your executor, and that you parse tool args with `json.loads(...)` as instructed.

---

## Check Before You Submit

- [ ] Filename is exact: `LastName_FirstName_csc7644_ca5_easy.py`
- [ ] `math.add` JSON Schema is implemented and strict:
  - `type: object`
  - `properties: a (integer), b (integer)`
  - `required: ["a", "b"]`
  - `additionalProperties: false`
- [ ] `exec_math_add` reads `a` and `b` and returns `{"sum": a + b}`
- [ ] Controller parses tool args from `call.function.arguments` using `json.loads(...)`
- [ ] Validation is called before executing tools (do not remove it)
- [ ] No API keys in source (use `.env`), and the `.env` file is not included in a Git repository
- [ ] Script runs the example goals without exceptions and prints a reasonable final answer
- [ ] PEP-8 style & comments for your edits (docstrings or inline comments for the parts you changed)
- [ ] Only allowed libraries used: `openai`, `python-dotenv`, `jsonschema` (plus stdlib)

---

## Submission Guidelines

Submit one Python file named `LastName_FirstName_csc7644_ca5_easy.py` to the Moodle submission link. Do not rename required functions or modes. Follow PEP-8, include docstrings for public functions, and add inline comments where logic isn't obvious. Avoid printing secrets; keep outputs minimal and deterministic where possible.

---
