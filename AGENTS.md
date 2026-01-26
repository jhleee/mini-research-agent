# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Research Agent - An intelligent research agent that decomposes complex queries into sub-questions, conducts web searches, reads webpages, and synthesizes findings into comprehensive markdown reports.

**Tech Stack:** Python with LangGraph for agent orchestration, LangChain/OpenAI for LLM calls, Textual for TUI, httpx for HTTP/SSE streaming.

**LLM:** Z.AI API (glm-4.7 model) with MCP tool endpoints for web search and reading.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

## Architecture

### LangGraph Workflow (`src/agent.py`)

The research agent follows this state machine:

```
plan_node → research_node → [conditional routing]
                              ├─ tools_node → process_results_node → research_node (loop)
                              └─ synthesize_node → END
```

- **plan_node**: Generates 3-5 research sub-queries via LLM
- **research_node**: Coordinates tool usage (web_search, read_webpage)
- **process_tool_results**: Extracts findings from tool responses
- **synthesize_node**: Creates final markdown report
- **should_continue**: Routes based on iteration count and tool calls

### State Management (`src/state.py`)

`ResearchState` TypedDict tracks: query, messages, research_plan, search_queries, read_urls, findings, iteration count, report, and status ("planning" → "researching" → "synthesizing" → "done").

### MCP Tool Integration (`src/tools.py`)

Two primary tools call Z.AI MCP endpoints:
- `web_search(query)` - Searches via webSearchPrime endpoint
- `read_webpage(url)` - Reads content via webReader endpoint

Uses HTTP streaming (SSE) with JSON-RPC 2.0 protocol. Content truncated to 8000 chars per page.

### UI Layer

**TUI (`src/tui/`)**: MVC-like architecture with Textual framework
- `controller.py` - ResearchController manages state and event callbacks
- `views.py` - ChatPanel, TaskPanel, InputBar widgets
- `models.py` - ChatMessage, Task, AppState dataclasses

## Configuration

Environment variables in `.env`:
```
OPENAI_API_KEY=<z-ai-api-key>
OPENAI_BASE_URL=https://api.z.ai/api/coding/paas/v4
OPENAI_MODEL=glm-4.7
```

Endpoints configured in `src/config.py`:
- `WEB_SEARCH_ENDPOINT` - Z.AI MCP web search
- `WEB_READER_ENDPOINT` - Z.AI MCP web reader
- `MAX_ITERATIONS = 10` - Research loop limit

## Key Entry Points

- `run.py` - Main entry point
- `src/agent.py:run_research_with_tools()` - Async API with event callbacks
