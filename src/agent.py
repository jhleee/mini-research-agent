"""Deep Research Agent orchestration layer.

This module provides the main API for running research with tools.
Core logic has been refactored into specialized modules:
- core.planning - Intent analysis, task decomposition, validation
- core.research - Research execution coordination
- core.synthesis - Report generation
- core.replanning - Dynamic replanning logic
- core.state_manager - State update and transition logic
- core.graph_builder - LangGraph graph construction
"""
import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI

from .state import ResearchState
from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from .tools import initialize_tools
from .core.graph_builder import create_research_graph


def create_llm():
    """Create the GLM LLM client."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY if OPENAI_API_KEY else "dummy",  # type: ignore
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL,
        temperature=0.7,
    )


async def run_research_with_tools(query: str, callback=None, config={"recursion_limit": 200}) -> str:
    """Run research with detailed event callbacks for tool usage.

    Args:
        query: The research question.
        callback: Callback function(event_type, data) for events.
        config: LangGraph configuration.

    Returns:
        The final research report.
    """
    tools = await initialize_tools()
    graph = create_research_graph(tools)

    initial_state = {
        "query": query,
        "messages": [],
        "hierarchical_plan": None,
        "research_plan": [],
        "current_main_task_id": None,
        "current_sub_task_id": None,
        "search_queries": [],
        "read_urls": [],
        "findings": [],
        "iteration": 0,
        "report": "",
        "status": "planning",
        "planning_phase": "intent_analysis",
        "needs_replanning": False,
        "pending_replan_request": None,
        "replan_count": 0,
    }

    def emit(event_type: str, data: dict):
        if callback:
            callback(event_type, data)

    final_state = None
    plan_emitted = False
    last_main_task_id = None
    last_sub_task_id = None

    async for state in graph.astream(initial_state, config=config):  # type: ignore
        final_state = state

        for node_name, node_state in state.items():
            emit("node_start", {"node": node_name})

            llm_nodes = {
                "plan": "Planning research",
                "research": "Analyzing",
                "replan": "Dynamic replanning",
                "synthesize": "Writing report",
            }
            if node_name in llm_nodes:
                emit("llm_start", {"message": llm_nodes[node_name]})

            if isinstance(node_state, dict):
                if not plan_emitted:
                    h_plan = node_state.get("hierarchical_plan")
                    if h_plan:
                        emit("hierarchical_plan_created", {
                            "intent_count": h_plan["intent_count"],
                            "main_tasks": [
                                {
                                    "id": mt["id"],
                                    "topic": mt["topic"],
                                    "sub_tasks": [
                                        {"id": st["id"], "query": st["query"]}
                                        for st in mt["sub_tasks"]
                                    ]
                                }
                                for mt in h_plan["main_tasks"]
                            ]
                        })
                        flat_queries = []
                        for mt in h_plan["main_tasks"]:
                            for st in mt["sub_tasks"]:
                                flat_queries.append(st["query"])
                        emit("plan_created", {"queries": flat_queries})
                        plan_emitted = True

                if node_name == "replan":
                    replan_request = node_state.get("pending_replan_request")
                    h_plan = node_state.get("hierarchical_plan")
                    if h_plan:
                        emit("dynamic_replan", {
                            "replan_count": node_state.get("replan_count", 0),
                            "extracted_items": replan_request.get("extracted_items", []) if replan_request else [],
                            "new_queries": [
                                st["query"]
                                for mt in h_plan["main_tasks"]
                                for st in mt["sub_tasks"]
                                if st["status"] == "pending"
                            ]
                        })

                current_main = node_state.get("current_main_task_id")
                current_sub = node_state.get("current_sub_task_id")

                if last_sub_task_id and (current_sub != last_sub_task_id):
                    emit("task_progress", {
                        "main_task_id": last_main_task_id,
                        "sub_task_id": last_sub_task_id,
                        "status": "completed"
                    })

                if current_main and current_sub and (current_sub != last_sub_task_id):
                    emit("task_progress", {
                        "main_task_id": current_main,
                        "sub_task_id": current_sub,
                        "status": "in_progress"
                    })
                    last_main_task_id = current_main
                    last_sub_task_id = current_sub

                search_queries = node_state.get("search_queries", [])
                if search_queries:
                    emit("query_completed", {"completed": search_queries})

                messages = node_state.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            emit("tool_call", {
                                "tool": tool_call.get("name", ""),
                                "args": tool_call.get("args", {}),
                            })

                    if hasattr(msg, "type") and msg.type == "tool":
                        emit("tool_result", {
                            "tool": getattr(msg, "name", ""),
                            "result": msg.content if hasattr(msg, "content") else "",
                        })

    if final_state:
        for node_state in final_state.values():
            if isinstance(node_state, dict) and node_state.get("report"):
                return node_state["report"]

    return "Research could not be completed."
