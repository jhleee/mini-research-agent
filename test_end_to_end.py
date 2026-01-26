"""End-to-End Test: Full Research Pipeline

Tests the complete research workflow from query to final report.
"""
import asyncio
import json
from src.agent import run_research_with_tools


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


class TestCallback:
    """Callback handler to track research progress."""

    def __init__(self):
        self.events = []
        self.plan_created = False
        self.hierarchical_plan = None
        self.queries_completed = []
        self.tool_calls = []
        self.tasks_in_progress = []
        self.tasks_completed = []

    def __call__(self, event_type: str, data: dict):
        """Handle events from the research agent."""
        self.events.append((event_type, data))

        if event_type == "hierarchical_plan_created":
            self.plan_created = True
            self.hierarchical_plan = data
            print(f"\n📋 Hierarchical Plan Created:")
            print(f"   Intent count: {data.get('intent_count')}")
            for mt in data.get("main_tasks", []):
                print(f"   Main Task [{mt['id']}]: {mt['topic']}")
                for st in mt.get("sub_tasks", []):
                    print(f"     └─ [{st['id']}] {st['query']}")

        elif event_type == "plan_created":
            queries = data.get("queries", [])
            print(f"\n📝 Flat Plan: {len(queries)} total queries")

        elif event_type == "task_progress":
            main_id = data.get("main_task_id")
            sub_id = data.get("sub_task_id")
            status = data.get("status")

            if status == "in_progress":
                self.tasks_in_progress.append(f"{main_id}.{sub_id}")
                print(f"\n▶️  Task [{main_id}.{sub_id}] started")
            elif status == "completed":
                self.tasks_completed.append(f"{main_id}.{sub_id}")
                print(f"\n✅ Task [{main_id}.{sub_id}] completed")

        elif event_type == "tool_call":
            tool = data.get("tool", "")
            args = data.get("args", {})
            self.tool_calls.append((tool, args))

            query = args.get("query", args.get("url", ""))
            print(f"\n🔧 Tool Call: {tool}")
            print(f"   Query: {query[:80]}...")

        elif event_type == "node_start":
            node = data.get("node", "")
            if node in ["plan", "research", "synthesize"]:
                print(f"\n🔄 Node: {node}")

        elif event_type == "llm_start":
            message = data.get("message", "")
            print(f"   💭 {message}...")


async def test_full_pipeline():
    """Test the complete research pipeline."""

    # Test query
    query = "부산과 순천의 인구 수를 비교해라"

    print_section(f"End-to-End Test: {query}")
    print("This will run the complete research pipeline.")
    print("Note: This test uses REAL API calls and may take some time.\n")

    # Create callback handler
    callback = TestCallback()

    # Run full research
    print_section("Starting Research Agent")

    try:
        report = await run_research_with_tools(
            query=query,
            callback=callback,
            config={"recursion_limit": 50}  # Lower limit for testing
        )

        # Results
        print_section("Research Complete!")

        print("Final Report Preview:")
        print("-" * 70)
        if len(report) > 500:
            print(report[:500] + "\n...\n[Report truncated - total length: {} chars]".format(len(report)))
        else:
            print(report)
        print("-" * 70)

        # Statistics
        print_section("Execution Statistics")

        print(f"Total events: {len(callback.events)}")
        print(f"Tool calls made: {len(callback.tool_calls)}")
        print(f"Tasks started: {len(callback.tasks_in_progress)}")
        print(f"Tasks completed: {len(callback.tasks_completed)}")

        # Detailed tool call breakdown
        if callback.tool_calls:
            print(f"\nTool calls breakdown:")
            web_searches = sum(1 for t, _ in callback.tool_calls if t == "web_search")
            web_reads = sum(1 for t, _ in callback.tool_calls if t == "read_webpage")
            print(f"  - web_search: {web_searches}")
            print(f"  - read_webpage: {web_reads}")

        # Task completion analysis
        if callback.hierarchical_plan:
            total_tasks = sum(
                len(mt.get("sub_tasks", []))
                for mt in callback.hierarchical_plan.get("main_tasks", [])
            )
            print(f"\nTask completion:")
            print(f"  - Total planned tasks: {total_tasks}")
            print(f"  - Tasks completed: {len(callback.tasks_completed)}")
            completion_rate = (len(callback.tasks_completed) / total_tasks * 100) if total_tasks > 0 else 0
            print(f"  - Completion rate: {completion_rate:.1f}%")

        # Analysis
        print_section("Test Analysis")

        print("Expected pipeline:")
        print("  1. ✅ Create hierarchical plan (2 main tasks: 부산, 순천)")
        print("  2. ✅ Execute searches for each sub-task")
        print("  3. ✅ Process tool results and advance tasks")
        print("  4. ✅ Synthesize findings into final report")

        print("\nActual results:")

        if callback.plan_created:
            intent_count = callback.hierarchical_plan.get("intent_count", 0)
            print(f"  ✅ Plan created with {intent_count} main topic(s)")
        else:
            print(f"  ❌ Plan not created")

        if callback.tool_calls:
            print(f"  ✅ Tools executed: {len(callback.tool_calls)} calls")
        else:
            print(f"  ⚠️  No tools executed")

        if callback.tasks_completed:
            print(f"  ✅ Tasks completed: {len(callback.tasks_completed)}")
        else:
            print(f"  ⚠️  No tasks completed")

        if report and len(report) > 100:
            print(f"  ✅ Report generated: {len(report)} chars")

            # Check if report mentions both cities
            has_busan = "부산" in report or "Busan" in report
            has_suncheon = "순천" in report or "Suncheon" in report

            if has_busan and has_suncheon:
                print(f"  ✅ Report covers both cities (부산, 순천)")
            else:
                print(f"  ⚠️  Report missing: 부산={has_busan}, 순천={has_suncheon}")
        else:
            print(f"  ❌ Report not generated or too short")

        print("\n" + "="*70)
        print("  Test Complete!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
