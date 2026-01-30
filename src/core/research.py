"""Research module for research agent.

Contains research execution logic:
- Research node coordination
- Task finding from plan
"""
from typing import Optional, Tuple, List, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from .state_manager import update_plan_findings
from ..state import ResearchState, HierarchicalPlan
from ..prompts import RESEARCHER_PROMPT


def find_current_task(
    plan: Optional[HierarchicalPlan],
    current_main_id: Optional[str],
    current_sub_id: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Find the current query and main topic from hierarchical plan."""
    if not plan or not current_main_id or not current_sub_id:
        return None, None

    for main_task in plan["main_tasks"]:
        if main_task["id"] == current_main_id:
            for sub_task in main_task["sub_tasks"]:
                if sub_task["id"] == current_sub_id:
                    return sub_task["query"], main_task["topic"]

    return None, None


def research_node(state: ResearchState, tools: Optional[List[BaseTool]] = None) -> dict:
    """Execute research using tools, with hierarchical task awareness."""
    llm = ChatOpenAI()
    if tools:
        llm = llm.bind_tools(tools)

    plan = state.get("hierarchical_plan")
    current_main_id = state.get("current_main_task_id")
    current_sub_id = state.get("current_sub_task_id")

    current_query = None
    current_main_topic = None

    if plan and current_main_id and current_sub_id:
        current_query, current_main_topic = find_current_task(plan, current_main_id, current_sub_id)

    if not current_query:
        flat_plan = state.get("research_plan", [])
        executed = state.get("search_queries", [])
        remaining = [q for q in flat_plan if q not in executed]

        if not remaining:
            return {
                "status": "synthesizing",
                "messages": [AIMessage(content="All planned searches completed. Moving to synthesis.")]
            }
        current_query = remaining[0]

    context_header = ""
    if current_main_topic:
        context_header = f"Main Research Topic: {current_main_topic}\n"

    findings_context = ""
    if state.get("findings"):
        findings_context = f"\n\nPrevious findings:\n" + "\n".join(state["findings"][-3:])

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"""{context_header}Research query: {state['query']}

Current search focus: {current_query}
{findings_context}

Use the web_search tool to search for: {current_query}
Then read relevant webpages to gather detailed information.""")
    ]

    messages.extend(state.get("messages", [])[-10:])

    response = llm.invoke(messages)
    executed = state.get("search_queries", [])

    return {
        "messages": [response],
        "search_queries": executed + [current_query] if current_query else executed,
        "iteration": state.get("iteration", 0) + 1,
    }
