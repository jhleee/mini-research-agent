"""Web search and reader tools using Z.AI MCP endpoints via langchain-mcp-adapters."""
import json
from typing import Optional
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import ZAI_API_KEY, WEB_SEARCH_ENDPOINT, WEB_READER_ENDPOINT

# Global MCP client instance
_mcp_client: Optional[MultiServerMCPClient] = None
_tools_cache: Optional[list] = None


def _get_mcp_client() -> MultiServerMCPClient:
    """Get or create the MCP client singleton."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MultiServerMCPClient({
            'zai-search': {
                'transport': 'streamable_http',
                'url': WEB_SEARCH_ENDPOINT,
                'headers': {
                    'Authorization': f'Bearer {ZAI_API_KEY}'
                }
            },
            'zai-reader': {
                'transport': 'streamable_http',
                'url': WEB_READER_ENDPOINT,
                'headers': {
                    'Authorization': f'Bearer {ZAI_API_KEY}'
                }
            }
        })
    return _mcp_client


async def get_mcp_tools() -> list:
    """Get all MCP tools from Z.AI servers."""
    global _tools_cache
    if _tools_cache is None:
        client = _get_mcp_client()
        _tools_cache = await client.get_tools()
    return _tools_cache


def extract_mcp_result(result) -> str:
    """Extract text content from MCP tool result."""
    if result is None:
        return "No results found."

    # Result is typically a list of content items
    if isinstance(result, list):
        texts = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                # Try to parse as JSON for structured results
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        # Format search results
                        formatted = []
                        for r in parsed[:10]:  # Limit to 10 results
                            title = r.get("title", "")
                            link = r.get("link", "")
                            content = r.get("content", "")
                            formatted.append(f"- {title}\n  {link}\n  {content[:200]}")
                        texts.append("\n\n".join(formatted))
                    else:
                        texts.append(str(parsed))
                except (json.JSONDecodeError, TypeError):
                    texts.append(text)
        return "\n".join(texts) if texts else "No results found."

    if isinstance(result, str):
        return result

    return str(result)


# Lazy-loaded tools - will be populated on first use
TOOLS = []


async def initialize_tools():
    """Initialize the tools list from MCP servers."""
    global TOOLS
    if not TOOLS:
        TOOLS.extend(await get_mcp_tools())
    return TOOLS
