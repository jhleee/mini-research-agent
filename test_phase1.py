"""Test script for Phase 1: Query Decomposition

Tests the hierarchical planning pipeline with a comparison query.
"""
import asyncio
import json
from src.agent import (
    create_llm,
    analyze_intent,
    decompose_task,
    build_hierarchical_plan
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


async def test_query_decomposition():
    """Test the query decomposition pipeline."""

    # Test query
    query = "부산과 순천의 인구 수를 비교해라"
    print_section(f"Input Query: {query}")

    # Create LLM
    llm = create_llm()

    # Phase 1: Intent Analysis
    print_section("Phase 1: Intent Analysis")
    intent_result = analyze_intent(query, llm)
    print("Intent Analysis Result:")
    print(json.dumps(intent_result, ensure_ascii=False, indent=2))

    # Phase 2: Task Decomposition (for each main topic)
    print_section("Phase 2: Task Decomposition")

    if intent_result.get("main_topics"):
        for idx, topic_info in enumerate(intent_result["main_topics"], 1):
            topic = topic_info.get("topic", "")
            description = topic_info.get("description", "")

            print(f"\n--- Main Topic {idx}: {topic} ---")
            print(f"Description: {description}")

            sub_tasks = decompose_task(topic, description, query, llm)
            print(f"\nGenerated Sub-tasks:")
            print(json.dumps(sub_tasks, ensure_ascii=False, indent=2))

    # Full Hierarchical Plan
    print_section("Full Hierarchical Plan")
    plan = build_hierarchical_plan(query, llm)

    print("Complete Plan Structure:")
    print(f"Intent Count: {plan['intent_count']}")
    print(f"Validated: {plan['is_validated']}")
    print(f"Refinements: {plan['refinement_count']}")

    print("\nMain Tasks:")
    for main_task in plan["main_tasks"]:
        print(f"\n  [{main_task['id']}] {main_task['topic']}")
        print(f"      Description: {main_task['description']}")
        print(f"      Sub-tasks:")
        for sub_task in main_task["sub_tasks"]:
            print(f"        - [{sub_task['id']}] {sub_task['query']}")

    # Analysis of results
    print_section("Analysis")

    print("Expected decomposition:")
    print("  1. 부산 인구 수를 찾는다 (search)")
    print("  2. 순천의 인구 수를 찾는다 (search)")
    print("  3. 두 결과를 비교한다 (synthesis)")

    print("\nActual decomposition:")
    total_queries = sum(len(mt["sub_tasks"]) for mt in plan["main_tasks"])
    print(f"  - Total main tasks: {len(plan['main_tasks'])}")
    print(f"  - Total sub-tasks (search queries): {total_queries}")

    print("\n✅ Key observations:")
    if plan["intent_count"] >= 2:
        print(f"  - System identified {plan['intent_count']} independent topics (부산, 순천)")
    else:
        print(f"  - System identified {plan['intent_count']} topic only")

    print(f"  - Generated {total_queries} total search queries")

    has_comparison = False
    for mt in plan["main_tasks"]:
        if "비교" in mt["topic"] or "comparison" in mt["topic"].lower():
            has_comparison = True
            break

    if has_comparison:
        print("  - System explicitly includes comparison task ✓")
    else:
        print("  - System does NOT explicitly include comparison task ✗")
        print("    (Comparison is expected to happen during synthesis phase)")


if __name__ == "__main__":
    asyncio.run(test_query_decomposition())
