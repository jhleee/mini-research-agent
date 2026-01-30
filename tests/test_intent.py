"""Direct test of analyze_intent."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.planning import analyze_intent
from src.agent import create_llm
from src.prompts import INTENT_ANALYZER_PROMPT

TEST_QUERY = "부산시와 순천시와 서귀포시의 인구 수를 비교해라"


async def test_analyze_intent():
    """Test intent analysis directly."""
    print(f"\nQuery: {TEST_QUERY}\n")

    llm = create_llm()
    result = analyze_intent(TEST_QUERY, llm)

    print(f"\n{'='*60}")
    print(f"Result:")
    print(f"{'='*60}")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(test_analyze_intent())
