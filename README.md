# AI Agents Apps — LangChain, LangGraph & MCP

Companion code for a book on building AI agent applications using LangChain, LangGraph, and the Model Context Protocol (MCP).

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (package manager)

## Setup

```bash
# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```
OPENAI_API_KEY=your-key-here
```

## Project Structure

```
chapters/
  ch01_intro_ai_agents/         # LangChain fundamentals
  ch02_executing_prompts_programmatically/  # Executing prompts with OpenAI & LangChain
```

## Running Notebooks

Chapters use [Marimo](https://marimo.io/) interactive notebooks:

```bash
marimo edit chapters/ch01_intro_ai_agents/langchain_fundamentals.py
marimo edit chapters/ch02_executing_prompts_programmatically/executing_prompt_prog.py
```

## Dependencies

- `langchain` — LangChain core
- `langchain-openai` — OpenAI integration for LangChain
- `openai` — OpenAI Python SDK
- `marimo` — Reactive notebook environment
