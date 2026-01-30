"""Test runner for research agent without TUI."""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent import run_research_with_tools
from src.config import OPENAI_API_KEY

# Test query
TEST_QUERY = "부산시와 순천시와 서귀포시의 인구 수를 비교해라"


def test_research_query():
    """Test the research agent with a specific query."""
    print(f"\n{'='*60}")
    print(f"Testing Research Agent")
    print(f"{'='*60}")
    print(f"\nQuery: {TEST_QUERY}")
    print(f"\n{'='*60}\n")

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set!")
        return

    async def run_test():
        """Run the research test."""
        plan_created = False
        tool_calls_made = []
        search_queries = []
        last_status = None

        def on_event(event_type: str, data: dict):
            """Handle events from research agent."""
            nonlocal plan_created, last_status

            print(f"\n[{event_type.upper()}]")

            if event_type == "node_start":
                node = data.get("node", "")
                print(f"  → Node: {node}")

            elif event_type == "hierarchical_plan_created":
                plan_created = True
                main_tasks = data.get("main_tasks", [])
                print(f"  ✓ Hierarchical plan created")
                print(f"  → Intent count: {data.get('intent_count', 0)}")
                print(f"  → Main tasks: {len(main_tasks)}")

                for mt in main_tasks:
                    print(f"\n  Main Task {mt['id']}: {mt['topic']}")
                    for st in mt.get("sub_tasks", []):
                        print(f"    - Sub Task {st['id']}: {st['query']}")

            elif event_type == "plan_created":
                queries = data.get("queries", [])
                print(f"  ✓ Plan created: {len(queries)} queries")

            elif event_type == "task_progress":
                main_id = data.get("main_task_id")
                sub_id = data.get("sub_task_id")
                status = data.get("status", "in_progress")
                print(f"  → Task {main_id}.{sub_id}: {status}")

            elif event_type == "tool_call":
                tool_name = data.get("tool", "")
                args = data.get("args", {})
                tool_calls_made.append(tool_name)

                if tool_name == "web_search":
                    query = args.get("search_query", args.get("query", ""))
                    search_queries.append(query)
                    print(f"  → Tool: {tool_name}")
                    print(f"    Query: {query}")

            elif event_type == "tool_result":
                tool_name = data.get("tool", "")
                result = data.get("result", "")
                print(f"  → Tool result: {tool_name}")
                if result and len(result) < 200:
                    print(f"    Result: {result[:100]}...")

            elif event_type == "llm_start":
                msg = data.get("message", "LLM responding")
                print(f"  → {msg}")

            elif event_type == "query_completed":
                completed = data.get("completed", [])
                print(f"  → Completed queries: {len(completed)}")

        # Run research
        print("Starting research...\n")

        try:
            report = await run_research_with_tools(
                TEST_QUERY,
                callback=on_event,
                config={"recursion_limit": 200}
            )

            print(f"\n{'='*60}")
            print("Research Complete!")
            print(f"{'='*60}")
            print(f"\nSummary:")
            print(f"  ✓ Hierarchical plan created: {plan_created}")
            print(f"  ✓ Unique search queries: {len(set(search_queries))}")
            print(f"  ✓ Total tool calls: {len(tool_calls_made)}")
            print(f"\n  Search queries made:")
            for i, query in enumerate(set(search_queries), 1):
                print(f"    {i}. {query}")

            if len(set(search_queries)) < 2:
                print(f"\n  ⚠ WARNING: Only {len(set(search_queries))} unique search query!")
                print(f"  Expected multiple diverse queries for the 3 cities.")

            print(f"\n{'='*60}")
            print("Final Report:")
            print(f"{'='*60}")
            print(f"\n{report}\n")

            # Validate
            success = (
                plan_created and
                len(set(search_queries)) >= 2 and
                "부산" in " ".join(search_queries) or
                "순천" in " ".join(search_queries) or
                "서귀포" in " ".join(search_queries)
            )

            if success:
                print(f"{'='*60}")
                print("✓ TEST PASSED")
                print(f"{'='*60}")
            else:
                print(f"{'='*60}")
                print("✗ TEST FAILED")
                print(f"{'='*60}")

        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Run the async test
    asyncio.run(run_test())


if __name__ == "__main__":
    test_research_query()
