"""Test script for Phase 2: Query Execution

Tests how the research_node executes search queries from the hierarchical plan.
"""
import asyncio
import json
from src.agent import (
    create_llm,
    build_hierarchical_plan,
    research_node,
    find_current_task,
)
from src.state import ResearchState


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


async def test_query_execution():
    """Test the query execution pipeline."""

    # Test query
    query = "부산과 순천의 인구 수를 비교해라"
    print_section(f"Input Query: {query}")

    # Create LLM
    llm = create_llm()

    # Build hierarchical plan (Phase 1)
    print_section("Phase 1: Building Hierarchical Plan")
    plan = build_hierarchical_plan(query, llm)

    print(f"Plan created with {plan['intent_count']} main tasks:")
    for mt in plan["main_tasks"]:
        print(f"  - [{mt['id']}] {mt['topic']}")
        for st in mt["sub_tasks"]:
            print(f"      - [{st['id']}] {st['query']}")

    # Simulate initial state after planning
    state: ResearchState = {
        "query": query,
        "messages": [],
        "hierarchical_plan": plan,
        "research_plan": [st["query"] for mt in plan["main_tasks"] for st in mt["sub_tasks"]],
        "current_main_task_id": "1",
        "current_sub_task_id": "1.1",
        "search_queries": [],
        "read_urls": [],
        "findings": [],
        "iteration": 0,
        "report": "",
        "status": "researching",
        "planning_phase": "complete",
    }

    # Phase 2: Execute first research iteration
    print_section("Phase 2: Executing First Research Query")

    # Get current task info
    current_query, current_topic = find_current_task(
        plan,
        state["current_main_task_id"],
        state["current_sub_task_id"]
    )

    print(f"Current Main Topic: {current_topic}")
    print(f"Current Sub-query: {current_query}")
    print(f"\nCalling research_node...\n")

    # Execute research node
    result = research_node(state)

    print_section("Research Node Result")
    print(f"Iteration: {result.get('iteration', 0)}")
    print(f"Search queries executed: {result.get('search_queries', [])}")

    # Check for messages
    messages = result.get("messages", [])
    if messages:
        print(f"\nMessages returned: {len(messages)}")
        for idx, msg in enumerate(messages, 1):
            print(f"\n--- Message {idx} ---")
            print(f"Type: {type(msg).__name__}")

            # Check for tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"Tool calls: {len(msg.tool_calls)}")
                for tc in msg.tool_calls:
                    print(f"  - Tool: {tc.get('name', 'unknown')}")
                    print(f"    Args: {json.dumps(tc.get('args', {}), ensure_ascii=False, indent=6)}")

            # Check content
            if hasattr(msg, "content"):
                content_preview = str(msg.content)[:200]
                print(f"Content preview: {content_preview}...")

    # Analysis
    print_section("Analysis")

    print("Expected behavior:")
    print("  1. research_node should identify current query: '부산 인구 2024'")
    print("  2. Should call web_search tool with this query")
    print("  3. Should return AIMessage with tool_calls")

    print("\nActual behavior:")
    if messages and hasattr(messages[0], "tool_calls") and messages[0].tool_calls:
        tool_call = messages[0].tool_calls[0]
        actual_tool = tool_call.get("name", "")
        actual_query = tool_call.get("args", {}).get("query", "")

        print(f"  ✅ Tool called: {actual_tool}")
        print(f"  ✅ Query used: {actual_query}")

        if "부산" in actual_query or "Busan" in actual_query:
            print(f"  ✅ Query correctly focuses on 부산 (Busan)")
        else:
            print(f"  ⚠️  Query does not focus on 부산")

    else:
        print(f"  ❌ No tool calls found in response")


if __name__ == "__main__":
    asyncio.run(test_query_execution())
