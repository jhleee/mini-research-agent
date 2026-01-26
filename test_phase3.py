"""Test script for Phase 3: Tool Execution and Result Processing

Tests how tools_node executes and process_tool_results handles the output.
"""
import asyncio
import json
from langchain_core.messages import AIMessage, ToolMessage
from src.agent import (
    create_llm,
    build_hierarchical_plan,
    process_tool_results,
    get_next_pending_task,
)
from src.state import ResearchState
from src.tools import TOOLS


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


async def test_tool_processing():
    """Test the tool execution and result processing pipeline."""

    # Test query
    query = "부산과 순천의 인구 수를 비교해라"
    print_section(f"Input Query: {query}")

    # Create LLM
    llm = create_llm()

    # Build hierarchical plan
    print_section("Building Hierarchical Plan")
    plan = build_hierarchical_plan(query, llm)

    print(f"Plan created with {plan['intent_count']} main tasks:")
    for mt in plan["main_tasks"]:
        print(f"  - [{mt['id']}] {mt['topic']}: {len(mt['sub_tasks'])} sub-tasks")

    # Simulate state after a tool call
    print_section("Phase 3: Simulating Tool Execution")

    # Create a mock AIMessage with tool call
    ai_message = AIMessage(
        content="",
        tool_calls=[{
            "name": "web_search",
            "args": {"query": "부산 인구수 2024"},
            "id": "call_123"
        }]
    )

    # Simulate tool result
    tool_result_message = ToolMessage(
        content="""부산광역시의 2024년 인구는 약 340만 명입니다.
부산은 서울 다음으로 인구가 많은 대도시이며, 최근 몇 년간 인구가 소폭 감소하는 추세를 보이고 있습니다.
2020년에는 약 342만 명이었으나 2024년 현재 340만 명 수준으로 집계되고 있습니다.

주요 구별 인구 분포:
- 해운대구: 약 42만 명
- 부산진구: 약 38만 명
- 사하구: 약 33만 명

참고: 행정안전부 주민등록인구통계 기준""",
        tool_call_id="call_123",
        name="web_search"
    )

    print("Simulated tool execution:")
    print(f"  - Tool: web_search")
    print(f"  - Query: 부산 인구수 2024")
    print(f"  - Result length: {len(tool_result_message.content)} chars")

    # Create state with tool results
    state: ResearchState = {
        "query": query,
        "messages": [ai_message, tool_result_message],
        "hierarchical_plan": plan,
        "research_plan": [st["query"] for mt in plan["main_tasks"] for st in mt["sub_tasks"]],
        "current_main_task_id": "1",
        "current_sub_task_id": "1.1",
        "search_queries": ["부산 인구수 2024"],
        "read_urls": [],
        "findings": [],
        "iteration": 1,
        "report": "",
        "status": "researching",
        "planning_phase": "complete",
    }

    # Process tool results
    print_section("Processing Tool Results")
    result = process_tool_results(state)

    print("Process results output:")
    print(f"  - New findings added: {len(result.get('findings', []))}")
    print(f"  - Next main task ID: {result.get('current_main_task_id')}")
    print(f"  - Next sub task ID: {result.get('current_sub_task_id')}")

    if result.get('findings'):
        print(f"\nFindings content preview:")
        for idx, finding in enumerate(result['findings'], 1):
            preview = finding[:100] + "..." if len(finding) > 100 else finding
            print(f"  {idx}. {preview}")

    # Check plan updates
    updated_plan = result.get('hierarchical_plan')
    if updated_plan:
        print(f"\nHierarchical plan status:")
        for mt in updated_plan["main_tasks"]:
            print(f"  Main Task [{mt['id']}]: {mt['status']}")
            for st in mt["sub_tasks"]:
                print(f"    Sub Task [{st['id']}]: {st['status']} - {len(st.get('findings', []))} findings")

    # Analysis
    print_section("Analysis")

    print("Expected behavior:")
    print("  1. Extract findings from tool result (부산 인구 정보)")
    print("  2. Update hierarchical plan: mark task 1.1 as COMPLETED")
    print("  3. Move to next pending task (1.2 or 2.1)")
    print("  4. Store findings for later synthesis")

    print("\nActual behavior:")

    if result.get('findings'):
        print(f"  ✅ Findings extracted: {len(result['findings'])} items")
        if "부산" in result['findings'][0] or "340" in result['findings'][0]:
            print(f"  ✅ Findings contain 부산 population data")
    else:
        print(f"  ❌ No findings extracted")

    if updated_plan:
        task_1_1 = updated_plan["main_tasks"][0]["sub_tasks"][0]
        if task_1_1["status"] == "completed":
            print(f"  ✅ Task 1.1 marked as completed")
        else:
            print(f"  ⚠️  Task 1.1 status: {task_1_1['status']}")

        if task_1_1.get("findings"):
            print(f"  ✅ Task 1.1 has {len(task_1_1['findings'])} findings stored")

    next_main, next_sub = result.get('current_main_task_id'), result.get('current_sub_task_id')
    if next_main and next_sub and (next_main != "1" or next_sub != "1.1"):
        print(f"  ✅ Advanced to next task: [{next_main}.{next_sub}]")
    elif next_main is None:
        print(f"  ✅ All tasks completed")
    else:
        print(f"  ⚠️  Still on task [{next_main}.{next_sub}]")


if __name__ == "__main__":
    asyncio.run(test_tool_processing())
