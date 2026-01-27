"""Web search and reader tools using Z.AI MCP endpoints."""
import httpx
import json
from typing import Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .config import OPENAI_API_KEY, WEB_SEARCH_ENDPOINT, WEB_READER_ENDPOINT


class SearchResult(BaseModel):
    """Search result schema."""
    title: str
    url: str
    snippet: str
    site_name: str = ""


class WebContent(BaseModel):
    """Web content schema."""
    title: str
    url: str
    content: str
    links: list[str] = Field(default_factory=list)


async def call_mcp_tool(endpoint: str, tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool endpoint using Streamable HTTP."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    # MCP JSON-RPC format
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(endpoint, json=payload, headers=headers)

        # Check if SSE response
        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            # Parse SSE response
            result = None
            for line in response.text.split("\n"):
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        try:
                            parsed = json.loads(data)
                            if "result" in parsed:
                                result = parsed["result"]
                            elif "error" in parsed:
                                raise Exception(f"MCP Error: {parsed['error']}")
                        except json.JSONDecodeError:
                            pass
            return result or {}
        else:
            # Regular JSON response
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise Exception(f"MCP Error: {result['error']}")

            return result.get("result", {})


async def call_mcp_sse(endpoint: str, tool_name: str, arguments: dict) -> str:
    """Call MCP endpoint with SSE streaming support."""
    # Check API key
    if not OPENAI_API_KEY:
        return "Error: API key not configured. Please set OPENAI_API_KEY in .env"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Try streaming first
        async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
            if response.status_code == 401:
                return "Error: Invalid API key (401 Unauthorized)"
            elif response.status_code == 403:
                return "Error: API access forbidden (403 Forbidden)"
            elif response.status_code != 200:
                # Read error body
                error_body = await response.aread()
                return f"Error: HTTP {response.status_code} - {error_body.decode('utf-8', errors='replace')[:500]}"

            content_type = response.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                # SSE streaming
                result_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data:
                            try:
                                parsed = json.loads(data)
                                if "result" in parsed:
                                    result_content = extract_content(parsed["result"])
                                elif "error" in parsed:
                                    return f"Error: {parsed['error']}"
                            except json.JSONDecodeError:
                                pass
                return result_content or "No results"
            else:
                # Regular JSON
                text = await response.aread()
                data = json.loads(text)
                if "error" in data:
                    return f"Error: {data['error']}"
                return extract_content(data.get("result", {}))


def extract_content(result: dict) -> str:
    """Extract text content from MCP result."""
    if not result:
        return "No results found."

    content = result.get("content", [])
    if not content:
        return str(result) if result else "No results found."

    # Extract text content
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            return item.get("text", "")

    return str(result)


@tool
async def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query to look up on the web.

    Returns:
        Search results with titles, URLs, and snippets.
    """
    try:
        result = await call_mcp_sse(
            WEB_SEARCH_ENDPOINT,
            "webSearchPrime",
            {"query": query}
        )
        return result if result else "No search results found."

    except Exception as e:
        return f"Search error: {str(e)}"


@tool
async def read_webpage(url: str) -> str:
    """Read and extract content from a webpage.

    Args:
        url: The URL of the webpage to read.

    Returns:
        The main content extracted from the webpage.
    """
    try:
        result = await call_mcp_sse(
            WEB_READER_ENDPOINT,
            "webReader",
            {"url": url}
        )

        # Truncate if too long
        if result and len(result) > 8000:
            result = result[:8000] + "\n\n[Content truncated...]"

        return result if result else "Could not read webpage content."

    except Exception as e:
        return f"Read error: {str(e)}"


# Tool list for the agent
TOOLS = [web_search, read_webpage]
