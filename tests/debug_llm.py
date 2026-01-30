"""Debug LLM responses."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent import create_llm
from src.prompts import INTENT_ANALYZER_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

TEST_QUERY = "부산시와 순천시와 서귀포시의 인구 수를 비교해라"


async def debug_llm():
    """Debug LLM intent analysis."""
    print(f"\nQuery: {TEST_QUERY}\n")

    llm = create_llm()
    messages = [
        SystemMessage(content=INTENT_ANALYZER_PROMPT),
        HumanMessage(content=f"Analyze this research query: {TEST_QUERY}")
    ]

    print("Sending to LLM...")
    response = await llm.ainvoke(messages)

    print(f"\n{'='*60}")
    print(f"Raw LLM Response:")
    print(f"{'='*60}")
    print(response.content)
    print(f"\n{'='*60}\n")

    # Try to parse
    from src.agent import parse_json_response
    import json

    parsed = parse_json_response(response.content)

    if parsed:
        print(f"Parsed JSON:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        print(f"parse_json_response failed")


if __name__ == "__main__":
    asyncio.run(debug_llm())
