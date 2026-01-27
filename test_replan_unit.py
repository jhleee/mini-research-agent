"""Unit Tests for Dynamic Replanning Feature

Tests the replanning logic without making actual API calls.
"""
import copy
from src.state import (
    ResearchState,
    HierarchicalPlan,
    MainTask,
    SubTask,
    TaskStatus,
    DynamicQueryRequest,
)
from src.agent import (
    add_dynamic_tasks_to_plan,
    get_next_pending_task,
    should_continue_after_process,
)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def create_test_plan() -> HierarchicalPlan:
    """Create a test hierarchical plan."""
    return {
        "original_query": "파스타 종류 3가지를 찾고 재료를 비교해줘",
        "intent_count": 1,
        "main_tasks": [
            {
                "id": "1",
                "topic": "파스타 종류",
                "description": "파스타 종류를 검색하고 각 재료를 비교",
                "status": TaskStatus.IN_PROGRESS.value,
                "sub_tasks": [
                    {
                        "id": "1.1",
                        "query": "파스타 종류 3가지",
                        "status": TaskStatus.COMPLETED.value,
                        "findings": ["스파게티, 펜네, 라자냐 발견"]
                    }
                ],
                "summary": ""
            }
        ],
        "is_validated": True,
        "refinement_count": 0
    }


def test_add_dynamic_tasks_to_plan():
    """Test adding dynamic tasks to an existing plan."""
    print_section("Test: add_dynamic_tasks_to_plan")

    # Create initial plan
    plan = create_test_plan()
    initial_task_count = len(plan["main_tasks"][0]["sub_tasks"])
    print(f"Initial sub-tasks: {initial_task_count}")

    # Dynamic queries to add
    dynamic_queries = [
        {"item": "스파게티", "query": "스파게티 핵심 재료", "purpose": "재료 파악"},
        {"item": "펜네", "query": "펜네 핵심 재료", "purpose": "재료 파악"},
        {"item": "라자냐", "query": "라자냐 핵심 재료", "purpose": "재료 파악"},
    ]

    # Add dynamic tasks
    updated_plan = add_dynamic_tasks_to_plan(plan, dynamic_queries, "1")

    # Verify
    final_task_count = len(updated_plan["main_tasks"][0]["sub_tasks"])
    print(f"Final sub-tasks: {final_task_count}")

    assert final_task_count == initial_task_count + len(dynamic_queries), \
        f"Expected {initial_task_count + len(dynamic_queries)} tasks, got {final_task_count}"

    # Check task IDs
    for idx, st in enumerate(updated_plan["main_tasks"][0]["sub_tasks"][1:], start=2):
        expected_id = f"1.{idx}"
        assert st["id"] == expected_id, f"Expected ID {expected_id}, got {st['id']}"
        assert st["status"] == TaskStatus.PENDING.value, f"New task should be PENDING"

    # Check queries
    new_queries = [st["query"] for st in updated_plan["main_tasks"][0]["sub_tasks"][1:]]
    expected_queries = [q["query"] for q in dynamic_queries]
    assert new_queries == expected_queries, f"Queries don't match"

    print("Sub-tasks after dynamic addition:")
    for st in updated_plan["main_tasks"][0]["sub_tasks"]:
        print(f"  [{st['id']}] {st['query']} - {st['status']}")

    print("\n✅ Test passed!")


def test_get_next_pending_task():
    """Test finding the next pending task."""
    print_section("Test: get_next_pending_task")

    plan = create_test_plan()

    # Add some pending tasks
    dynamic_queries = [
        {"item": "스파게티", "query": "스파게티 재료", "purpose": ""},
        {"item": "펜네", "query": "펜네 재료", "purpose": ""},
    ]
    plan = add_dynamic_tasks_to_plan(plan, dynamic_queries, "1")

    # First pending should be 1.2
    main_id, sub_id = get_next_pending_task(plan)
    print(f"Next pending task: {main_id}.{sub_id}")

    assert main_id == "1", f"Expected main_id '1', got '{main_id}'"
    assert sub_id == "1.2", f"Expected sub_id '1.2', got '{sub_id}'"

    # Mark 1.2 as completed
    for st in plan["main_tasks"][0]["sub_tasks"]:
        if st["id"] == "1.2":
            st["status"] = TaskStatus.COMPLETED.value
            break

    # Now next pending should be 1.3
    main_id, sub_id = get_next_pending_task(plan)
    print(f"After completing 1.2, next: {main_id}.{sub_id}")

    assert sub_id == "1.3", f"Expected sub_id '1.3', got '{sub_id}'"

    print("\n✅ Test passed!")


def test_should_continue_after_process():
    """Test the routing decision after processing tool results."""
    print_section("Test: should_continue_after_process")

    # Test case 1: No replanning needed
    state1: ResearchState = {
        "query": "test",
        "messages": [],
        "research_plan": [],
        "search_queries": [],
        "read_urls": [],
        "findings": [],
        "iteration": 0,
        "report": "",
        "status": "researching",
        "hierarchical_plan": None,
        "current_main_task_id": None,
        "current_sub_task_id": None,
        "planning_phase": "complete",
        "needs_replanning": False,
        "pending_replan_request": None,
        "replan_count": 0,
    }

    result1 = should_continue_after_process(state1)
    print(f"Case 1 (no replanning): {result1}")
    assert result1 == "research", f"Expected 'research', got '{result1}'"

    # Test case 2: Replanning needed
    state2 = copy.deepcopy(state1)
    state2["needs_replanning"] = True
    state2["pending_replan_request"] = {
        "trigger_query": "파스타 종류 3가지",
        "extracted_items": ["스파게티", "펜네", "라자냐"],
        "query_template": "{item} 핵심 재료",
        "purpose": "각 파스타 재료 파악"
    }

    result2 = should_continue_after_process(state2)
    print(f"Case 2 (replanning needed): {result2}")
    assert result2 == "replan", f"Expected 'replan', got '{result2}'"

    # Test case 3: needs_replanning True but no request
    state3 = copy.deepcopy(state1)
    state3["needs_replanning"] = True
    state3["pending_replan_request"] = None

    result3 = should_continue_after_process(state3)
    print(f"Case 3 (flag True, no request): {result3}")
    assert result3 == "research", f"Expected 'research', got '{result3}'"

    print("\n✅ Test passed!")


def test_dynamic_query_request_structure():
    """Test the DynamicQueryRequest structure."""
    print_section("Test: DynamicQueryRequest Structure")

    request: DynamicQueryRequest = {
        "trigger_query": "파스타 종류 3가지",
        "extracted_items": ["스파게티", "펜네", "라자냐"],
        "query_template": "{item} 핵심 재료",
        "purpose": "각 파스타의 핵심 재료를 검색"
    }

    print(f"Trigger query: {request['trigger_query']}")
    print(f"Extracted items: {request['extracted_items']}")
    print(f"Query template: {request['query_template']}")
    print(f"Purpose: {request['purpose']}")

    # Generate queries from template
    generated_queries = [
        request["query_template"].replace("{item}", item)
        for item in request["extracted_items"]
    ]

    print(f"\nGenerated queries:")
    for q in generated_queries:
        print(f"  • {q}")

    expected = ["스파게티 핵심 재료", "펜네 핵심 재료", "라자냐 핵심 재료"]
    assert generated_queries == expected, f"Queries don't match expected"

    print("\n✅ Test passed!")


def test_graph_structure():
    """Test that the graph has the correct structure."""
    print_section("Test: Graph Structure")

    from src.agent import create_research_graph

    graph = create_research_graph()
    nodes = list(graph.nodes.keys())

    print(f"Graph nodes: {nodes}")

    # Check required nodes exist
    required_nodes = ["plan", "research", "tools", "process_results", "replan", "synthesize"]
    for node in required_nodes:
        assert node in nodes, f"Missing node: {node}"
        print(f"  ✓ {node}")

    print("\n✅ Test passed!")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("  DYNAMIC REPLANNING UNIT TESTS")
    print("=" * 60)

    tests = [
        test_dynamic_query_request_structure,
        test_add_dynamic_tasks_to_plan,
        test_get_next_pending_task,
        test_should_continue_after_process,
        test_graph_structure,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
