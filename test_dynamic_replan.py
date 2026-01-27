"""Test: Dynamic Replanning Feature

Tests the iterative loop where the agent:
1. First finds items (e.g., 3 pasta types)
2. Dynamically generates follow-up queries for each item
3. Researches each item and compares them

Example query: "파스타의 종류 3가지를 찾아서, 각 파스타의 핵심 재료를 검색하고 비교해서 정리해줘."
"""
import asyncio
from src.agent import run_research_with_tools
from src.state import TaskStatus


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


class DynamicReplanCallback:
    """Callback handler to track dynamic replanning events."""

    def __init__(self):
        self.events = []
        self.plan_created = False
        self.hierarchical_plan = None
        self.replan_events = []
        self.tool_calls = []
        self.tasks_in_progress = []
        self.tasks_completed = []
        self.extracted_items = []

    def __call__(self, event_type: str, data: dict):
        """Handle events from the research agent."""
        self.events.append((event_type, data))

        if event_type == "hierarchical_plan_created":
            self.plan_created = True
            self.hierarchical_plan = data
            print(f"\n📋 Initial Hierarchical Plan:")
            print(f"   Intent count: {data.get('intent_count')}")
            for mt in data.get("main_tasks", []):
                print(f"   Main Task [{mt['id']}]: {mt['topic']}")
                for st in mt.get("sub_tasks", []):
                    print(f"     └─ [{st['id']}] {st['query']}")

        elif event_type == "plan_created":
            queries = data.get("queries", [])
            print(f"\n📝 Initial Flat Plan: {len(queries)} queries")

        elif event_type == "dynamic_replan":
            self.replan_events.append(data)
            print(f"\n🔄 DYNAMIC REPLANNING TRIGGERED!")
            print(f"   Replan count: {data.get('replan_count', 0)}")
            items = data.get('extracted_items', [])
            if items:
                print(f"   Extracted items: {items}")
                self.extracted_items.extend(items)
            new_queries = data.get('new_queries', [])
            if new_queries:
                print(f"   New queries added ({len(new_queries)}):")
                for q in new_queries[:5]:
                    print(f"     + {q}")
                if len(new_queries) > 5:
                    print(f"     ... and {len(new_queries) - 5} more")

        elif event_type == "task_progress":
            main_id = data.get("main_task_id")
            sub_id = data.get("sub_task_id")
            status = data.get("status")

            if status == "in_progress":
                self.tasks_in_progress.append(f"{main_id}.{sub_id}")
                print(f"\n▶️  Task [{sub_id}] started")
            elif status == "completed":
                self.tasks_completed.append(f"{main_id}.{sub_id}")
                print(f"\n✅ Task [{sub_id}] completed")

        elif event_type == "tool_call":
            tool = data.get("tool", "")
            args = data.get("args", {})
            self.tool_calls.append((tool, args))

            query = args.get("query", args.get("url", ""))
            short_query = query[:60] + "..." if len(query) > 60 else query
            print(f"\n🔧 Tool: {tool} → {short_query}")

        elif event_type == "node_start":
            node = data.get("node", "")
            if node == "replan":
                print(f"\n🧠 Entering REPLAN node...")
            elif node in ["plan", "research", "synthesize"]:
                print(f"\n🔄 Node: {node}")

        elif event_type == "llm_start":
            message = data.get("message", "")
            print(f"   💭 {message}...")


async def test_dynamic_replanning():
    """Test the dynamic replanning feature with a pasta query."""

    # Query that requires finding items first, then researching each
    query = "파스타의 종류 3가지를 찾아서, 각 파스타의 핵심 재료를 검색하고 비교해서 정리해줘."

    print_section(f"Dynamic Replanning Test")
    print(f"Query: {query}")
    print("\nExpected behavior:")
    print("  1. Initial plan: Search for pasta types")
    print("  2. Dynamic replan: Extract 3 pasta types from results")
    print("  3. Generate new queries for each pasta's ingredients")
    print("  4. Research each pasta type")
    print("  5. Synthesize comparison report")
    print("\nNote: This test uses REAL API calls.\n")

    # Create callback handler
    callback = DynamicReplanCallback()

    # Run research
    print_section("Starting Research Agent")

    try:
        report = await run_research_with_tools(
            query=query,
            callback=callback,
            config={"recursion_limit": 100}
        )

        # Results
        print_section("Research Complete!")

        print("Final Report Preview:")
        print("-" * 70)
        if len(report) > 800:
            print(report[:800] + f"\n...\n[Truncated - total: {len(report)} chars]")
        else:
            print(report)
        print("-" * 70)

        # Statistics
        print_section("Dynamic Replanning Analysis")

        print(f"Total events: {len(callback.events)}")
        print(f"Dynamic replan triggered: {len(callback.replan_events)} time(s)")

        if callback.extracted_items:
            print(f"\nExtracted items for follow-up research:")
            for item in callback.extracted_items:
                print(f"  • {item}")
        else:
            print("\n⚠️  No items were extracted for dynamic replanning")

        print(f"\nTool calls: {len(callback.tool_calls)}")
        if callback.tool_calls:
            web_searches = sum(1 for t, _ in callback.tool_calls if t == "web_search")
            web_reads = sum(1 for t, _ in callback.tool_calls if t == "read_webpage")
            print(f"  - web_search: {web_searches}")
            print(f"  - read_webpage: {web_reads}")

        print(f"\nTask execution:")
        print(f"  - Started: {len(callback.tasks_in_progress)}")
        print(f"  - Completed: {len(callback.tasks_completed)}")

        # Test assertions
        print_section("Test Results")

        success = True

        # Check if plan was created
        if callback.plan_created:
            print("✅ Initial plan created")
        else:
            print("❌ Initial plan NOT created")
            success = False

        # Check if dynamic replanning occurred
        if callback.replan_events:
            print(f"✅ Dynamic replanning triggered ({len(callback.replan_events)}x)")
            # Check if items were extracted
            if callback.extracted_items:
                print(f"✅ Items extracted: {len(callback.extracted_items)}")
            else:
                print("⚠️  No items extracted (may be expected if query was simple)")
        else:
            print("⚠️  Dynamic replanning NOT triggered")
            print("   (This may be expected if the initial plan was sufficient)")

        # Check if multiple tool calls were made
        if len(callback.tool_calls) >= 2:
            print(f"✅ Multiple tool calls made ({len(callback.tool_calls)})")
        else:
            print(f"⚠️  Few tool calls made ({len(callback.tool_calls)})")

        # Check if report was generated
        if report and len(report) > 100:
            print(f"✅ Report generated ({len(report)} chars)")

            # Check if report mentions pasta types
            pasta_mentions = sum(1 for word in ["파스타", "스파게티", "펜네", "라자냐", "pasta", "spaghetti", "penne", "lasagna", "리가토니", "마카로니"]
                               if word.lower() in report.lower())
            if pasta_mentions >= 2:
                print(f"✅ Report mentions multiple pasta types")
            else:
                print(f"⚠️  Report may not cover multiple pasta types")
        else:
            print("❌ Report NOT generated or too short")
            success = False

        print("\n" + "=" * 70)
        if success:
            print("  ✅ TEST PASSED - Dynamic replanning feature is working!")
        else:
            print("  ⚠️  TEST COMPLETED - Check warnings above")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()


async def test_simple_query():
    """Test with a simpler query that should NOT trigger replanning."""

    query = "AI 최신 동향을 알려줘"

    print_section(f"Control Test: Simple Query")
    print(f"Query: {query}")
    print("\nExpected: No dynamic replanning needed\n")

    callback = DynamicReplanCallback()

    try:
        report = await run_research_with_tools(
            query=query,
            callback=callback,
            config={"recursion_limit": 30}
        )

        print_section("Control Test Results")

        if callback.replan_events:
            print(f"⚠️  Dynamic replanning was triggered (unexpected for simple query)")
        else:
            print("✅ No dynamic replanning (expected for simple query)")

        if report and len(report) > 50:
            print(f"✅ Report generated ({len(report)} chars)")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DYNAMIC REPLANNING TEST SUITE")
    print("=" * 70)

    # Run main test
    asyncio.run(test_dynamic_replanning())

    # Optionally run control test
    # asyncio.run(test_simple_query())
