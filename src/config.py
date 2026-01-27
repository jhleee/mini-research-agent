"""Configuration settings for the research agent."""
import os
from dotenv import load_dotenv

load_dotenv()

# GLM API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "glm-4.7")

# Z.AI MCP Endpoints
WEB_SEARCH_ENDPOINT = "https://api.z.ai/api/mcp/web_search_prime/mcp"
WEB_READER_ENDPOINT = "https://api.z.ai/api/mcp/web_reader/mcp"

# Agent Configuration
MAX_SEARCH_RESULTS = 5
MAX_ITERATIONS = 10

# Planning Configuration (Hierarchical Task Decomposition)
MAX_MAIN_TASKS = 4       # Maximum independent topics from intent analysis
MAX_SUB_TASKS = 4        # Maximum queries per main task
MAX_REFINEMENTS = 1      # Maximum plan refinement iterations
VALIDATION_THRESHOLD = 0.7  # Confidence threshold for plan validation

# Dynamic Replanning Configuration
MAX_REPLANS = 3          # Maximum dynamic replanning iterations per research session
MAX_DYNAMIC_ITEMS = 5    # Maximum items to extract for follow-up queries
