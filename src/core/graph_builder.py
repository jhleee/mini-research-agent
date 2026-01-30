"""Graph builder module for research agent.

Contains graph construction logic:
- Routing conditions
- Graph node connection
- Research graph creation
"""
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from ..state import ResearchState
from .research import research_node
from .synthesis import synthesize_node
from .replanning import add_dynamic_tasks_to_plan
from .state_manager import process_tool_results


def replan_node(state: ResearchState) -> dict:
    """Process dynamic replanning request and add new sub-tasks to the plan."""
    from .replanning import generate_dynamic_queries, add_dynamic_tasks_to_plan
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI()
    replan_request = state.get("pending_replan_request")
    plan = state.get("hierarchical_plan")
    current_main_id = state.get("current_main_task_id")

    if not replan_request or not plan:
        return {
            "needs_replanning": False,
            "pending_replan_request": None,
        }

    target_main_id = current_main_id or "1"

    dynamic_queries = generate_dynamic_queries(
        state["query"],
        replan_request,
        llm
    )

    if not dynamic_queries:
        return {
            "needs_replanning": False,
            "pending_replan_request": None,
        }

    updated_plan = add_dynamic_tasks_to_plan(
        plan,
        dynamic_queries,
        target_main_id
    )

    flat_plan = state.get("research_plan", [])[:]
    for query_info in dynamic_queries:
        query = query_info.get("query", "")
        if query and query not in flat_plan:
            flat_plan.append(query)

    from .state_manager import get_next_pending_task
    next_main_id, next_sub_id = get_next_pending_task(updated_plan)

    return {
        "hierarchical_plan": updated_plan,
        "research_plan": flat_plan,
        "current_main_task_id": next_main_id,
        "current_sub_task_id": next_sub_id,
        "needs_replanning": False,
        "pending_replan_request": None,
        "replan_count": state.get("replan_count", 0) + 1,
    }


def should_continue(state: ResearchState) -> Literal["research", "tools", "synthesize", "end"]:
    """Determine the next step in the workflow."""
    status = state.get("status", "planning")

    if status == "done":
        return "end"

    if status == "synthesizing":
        return "synthesize"

    if state.get("iteration", 0) >= 10:
        return "synthesize"

    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

    return "research"


def should_continue_after_process(state: ResearchState) -> Literal["research", "replan"]:
    """Determine whether to continue research or do dynamic replanning."""
    if state.get("needs_replanning", False) and state.get("pending_replan_request"):
        return "replan"

    return "research"


def create_research_graph(tools: list):
    """Create the research agent graph with dynamic replanning support."""
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan_node)  # type: ignore
    graph.add_node("research", lambda s: research_node(s, tools))  # type: ignore
    graph.add_node("tools", ToolNode(tools))  # type: ignore
    graph.add_node("process_results", process_tool_results)  # type: ignore
    graph.add_node("replan", replan_node)  # type: ignore
    graph.add_node("synthesize", synthesize_node)  # type: ignore

    graph.set_entry_point("plan")

    graph.add_edge("plan", "research")

    graph.add_conditional_edges(
        "research",
        should_continue,
        {
            "tools": "tools",
            "research": "research",
            "synthesize": "synthesize",
            "end": END,
        }
    )

    graph.add_edge("tools", "process_results")

    graph.add_conditional_edges(
        "process_results",
        should_continue_after_process,
        {
            "research": "research",
            "replan": "replan",
        }
    )

    graph.add_edge("replan", "research")
    graph.add_edge("synthesize", END)

    return graph.compile()


def plan_node(state: ResearchState) -> dict:
    """Create a hierarchical research plan from the query using agentic planning."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage
    from .planning import build_hierarchical_plan, get_first_pending_task

    llm = ChatOpenAI()
    query = state["query"]

    plan = build_hierarchical_plan(query, llm)

    flat_plan = []
    for main_task in plan["main_tasks"]:
        for sub_task in main_task["sub_tasks"]:
            flat_plan.append(sub_task["query"])

    first_main_id, first_sub_id = get_first_pending_task(plan)

    task_summary = []
    for mt in plan["main_tasks"]:
        task_summary.append(f"[{mt['topic']}]: {len(mt['sub_tasks'])} queries")

    return {
        "hierarchical_plan": plan,
        "research_plan": flat_plan,
        "current_main_task_id": first_main_id,
        "current_sub_task_id": first_sub_id,
        "search_queries": [],
        "read_urls": [],
        "findings": [],
        "iteration": 0,
        "status": "researching",
        "planning_phase": "complete",
        "messages": [AIMessage(content=f"Hierarchical plan created: {plan['intent_count']} main tasks - {', '.join(task_summary)}")]
    }
