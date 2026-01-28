"""Web search and reader tools using Z.AI MCP endpoints via langchain-mcp-adapters."""
import json
from typing import Optional
from functools import partial
from urllib.parse import urlsplit, urlunsplit, quote
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool

from .config import ZAI_API_KEY, WEB_SEARCH_ENDPOINT, WEB_READER_ENDPOINT

# Default parameters for webReader
WEB_READER_DEFAULTS = {
    "return_format": "markdown",
    "retain_images": False,
    "no_gfm": False,
    "with_images_summary": False,
    "with_links_summary": True,
}

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
                            formatted.append(
                                f"- {title}\n  {link}\n  {content[:200]}")
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
_original_web_reader = None


def _encode_url(url: str) -> str:
    """Encode URL to handle non-ASCII characters (e.g., Korean)."""
    parts = urlsplit(url)
    # Encode the path component, preserving slashes
    encoded_path = quote(parts.path, safe='/:@')
    # Encode the query component if present
    encoded_query = quote(parts.query, safe='=&') if parts.query else ''
    # Reconstruct the URL
    return urlunsplit((parts.scheme, parts.netloc, encoded_path, encoded_query, parts.fragment))


def _create_wrapped_web_reader(original_tool):
    """Create a wrapped webReader tool with default parameters."""
    global _original_web_reader
    _original_web_reader = original_tool

    async def web_reader_with_defaults(url: str) -> list:
        """Read and extract content from a webpage.

        Args:
            url: The URL of the webpage to read.

        Returns:
            The main content extracted from the webpage in markdown format.
        """
        # Encode URL to handle non-ASCII characters
        encoded_url = _encode_url(url)
        # Build input with defaults
        input_dict = {"url": encoded_url, **WEB_READER_DEFAULTS}
        return await _original_web_reader.ainvoke(input_dict)

    return StructuredTool.from_function(
        coroutine=web_reader_with_defaults,
        name="webReader",
        description=original_tool.description,
    )


async def initialize_tools():
    """Initialize the tools list from MCP servers."""
    global TOOLS
    if not TOOLS:
        mcp_tools = await get_mcp_tools()
        for tool in mcp_tools:
            if tool.name == "webReader":
                tool = _create_wrapped_web_reader(tool)
            TOOLS.append(tool)
    return TOOLS
