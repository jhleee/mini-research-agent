"""Replanning module for research agent.

Contains dynamic replanning logic:
- Analyze if replanning is needed
- Generate dynamic queries
- Add dynamic tasks to plan
"""
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import HierarchicalPlan, DynamicQueryRequest
from ..prompts import REPLAN_ANALYZER_PROMPT, DYNAMIC_QUERY_GENERATOR_PROMPT


def analyze_for_replanning(
    original_query: str,
    current_query: str,
    search_results: str,
    llm: ChatOpenAI
) -> Optional[DynamicQueryRequest]:
    """Analyze search results to determine if dynamic replanning is needed."""
    prompt = REPLAN_ANALYZER_PROMPT.format(
        original_query=original_query,
        current_query=current_query,
        search_results=search_results[:3000]
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Analyze these search results for dynamic replanning needs.")
    ]

    try:
        response = llm.invoke(messages)
        from .planning import parse_json_response
        result = parse_json_response(str(response.content))

        if not isinstance(result, dict):
            return None

        if not result.get("needs_replanning", False):
            return None

        extracted_items = result.get("extracted_items", [])
        if not extracted_items or not isinstance(extracted_items, list):
            return None

        from ..config import MAX_DYNAMIC_ITEMS
        extracted_items = extracted_items[:MAX_DYNAMIC_ITEMS]

        return {
            "trigger_query": current_query,
            "extracted_items": extracted_items,
            "query_template": str(result.get("query_template", "")),
            "purpose": str(result.get("purpose", ""))
        }

    except Exception:
        return None


def generate_dynamic_queries(
    original_query: str,
    replan_request: DynamicQueryRequest,
    llm: ChatOpenAI
) -> list[dict]:
    """Generate specific search queries for each extracted item."""
    prompt = DYNAMIC_QUERY_GENERATOR_PROMPT.format(
        original_query=original_query,
        items=", ".join(replan_request["extracted_items"]),
        purpose=replan_request["purpose"],
        template=replan_request["query_template"]
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Generate follow-up search queries for each item.")
    ]

    try:
        response = llm.invoke(messages)
        from .planning import parse_json_response
        result = parse_json_response(str(response.content))

        if not isinstance(result, dict):
            return [
                {
                    "item": item,
                    "query": replan_request["query_template"].replace("{item}", item),
                    "purpose": replan_request["purpose"]
                }
                for item in replan_request["extracted_items"]
            ]

        queries = result.get("queries", [])
        if not queries or not isinstance(queries, list):
            return [
                {
                    "item": item,
                    "query": replan_request["query_template"].replace("{item}", item),
                    "purpose": replan_request["purpose"]
                }
                for item in replan_request["extracted_items"]
            ]

        return queries

    except Exception:
        return [
            {
                "item": item,
                "query": replan_request["query_template"].replace("{item}", item),
                "purpose": replan_request["purpose"]
            }
            for item in replan_request["extracted_items"]
        ]


def add_dynamic_tasks_to_plan(
    plan: HierarchicalPlan,
    dynamic_queries: list[dict],
    parent_main_task_id: str
) -> HierarchicalPlan:
    """Add dynamically generated queries as new sub-tasks to the plan."""
    import copy
    plan = copy.deepcopy(plan)

    for main_task in plan["main_tasks"]:
        if main_task["id"] == parent_main_task_id:
            existing_count = len(main_task["sub_tasks"])

            for idx, query_info in enumerate(dynamic_queries, start=1):
                new_sub_task = {
                    "id": f"{parent_main_task_id}.{existing_count + idx}",
                    "query": str(query_info.get("query", ""))[:200],
                    "status": "pending",
                    "findings": []
                }
                main_task["sub_tasks"].append(new_sub_task)

            if main_task["status"] == "completed":
                main_task["status"] = "in_progress"

            break

    return plan
