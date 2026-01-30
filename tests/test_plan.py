"""Quick test for hierarchical planning only."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.planning import build_hierarchical_plan
from src.agent import create_llm
from src.config import OPENAI_API_KEY

TEST_QUERY = "부산시와 순천시와 서귀포시의 인구 수를 비교해라"


def test_planning():
    """Test hierarchical planning."""
    print(f"\n{'='*60}")
    print("Testing Hierarchical Planning")
    print(f"{'='*60}")
    print(f"\nQuery: {TEST_QUERY}\n")

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set!")
        return

    llm = create_llm()

    print("Building hierarchical plan...\n")
    print(f"LLM Model: {llm.model_name if hasattr(llm, 'model_name') else 'unknown'}")
    print(f"Temperature: {llm.temperature if hasattr(llm, 'temperature') else 'unknown'}\n")

    # First test analyze_intent directly
    from src.core.planning import analyze_intent
    print("Testing analyze_intent directly...")
    intent_result = analyze_intent(TEST_QUERY, llm)
    import json
    print(f"Intent result: {json.dumps(intent_result, ensure_ascii=False, indent=2)}\n")

    plan = build_hierarchical_plan(TEST_QUERY, llm)

    print(f"✓ Plan created!")
    print(f"  → Intent count: {plan['intent_count']}")
    print(f"  → Main tasks: {len(plan['main_tasks'])}")

    all_queries = []
    for mt in plan["main_tasks"]:
        print(f"\n  Main Task {mt['id']}: {mt['topic']}")
        print(f"    Description: {mt['description']}")
        print(f"    Sub tasks: {len(mt['sub_tasks'])}")

        for st in mt["sub_tasks"]:
            print(f"      - Sub Task {st['id']}: {st['query']}")
            all_queries.append(st['query'])

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"  Total main tasks: {len(plan['main_tasks'])}")
    print(f"  Total sub tasks: {sum(len(mt['sub_tasks']) for mt in plan['main_tasks'])}")
    print(f"  Unique search queries: {len(set(all_queries))}")

    print(f"\n  All queries:")
    for i, q in enumerate(all_queries, 1):
        print(f"    {i}. {q}")

    # Validation
    print(f"\n{'='*60}")
    print("Validation:")
    print(f"{'='*60}")

    issues = []

    if plan['intent_count'] != 3:
        issues.append(f"Expected 3 independent intents (3 cities), got {plan['intent_count']}")

    if len(all_queries) < 2:
        issues.append(f"Only {len(all_queries)} sub-tasks, expected at least 2")

    unique_queries = set(all_queries)
    if len(unique_queries) < 2:
        issues.append(f"Only {len(unique_queries)} unique queries - query decomposition failed!")

    expected_cities = ["부산", "순천", "서귀포"]
    for city in expected_cities:
        found = any(city in q for q in all_queries)
        if not found:
            issues.append(f"No queries found for '{city}'")

    if issues:
        print(f"  ✗ FAILED:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print(f"  ✓ PASSED:")
        print(f"    - All 3 cities identified as independent intents")
        print(f"    - Multiple sub-tasks created ({len(all_queries)} total)")
        print(f"    - Unique queries generated ({len(unique_queries)} distinct)")
        print(f"    - All cities covered in queries")
        return True


if __name__ == "__main__":
    success = test_planning()
    sys.exit(0 if success else 1)
