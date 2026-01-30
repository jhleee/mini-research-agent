"""State manager module for research agent.

Contains state management logic:
- Update plan with findings
- Get next pending task
- Process tool results
"""
import copy
from typing import Tuple, Optional

from ..state import ResearchState, HierarchicalPlan, TaskStatus
from ..config import MAX_REPLANS


def update_plan_findings(
    plan: HierarchicalPlan,
    main_id: str,
    sub_id: str,
    findings: list[str]
) -> HierarchicalPlan:
    """Update the hierarchical plan with new findings for a sub-task."""
    plan = copy.deepcopy(plan)

    for main_task in plan["main_tasks"]:
        if main_task["id"] == main_id:
            for sub_task in main_task["sub_tasks"]:
                if sub_task["id"] == sub_id:
                    sub_task["findings"].extend(findings)
                    sub_task["status"] = TaskStatus.COMPLETED.value
                    break

            if main_task["status"] == TaskStatus.PENDING.value:
                main_task["status"] = TaskStatus.IN_PROGRESS.value

            if all(st["status"] == TaskStatus.COMPLETED.value for st in main_task["sub_tasks"]):
                main_task["status"] = TaskStatus.COMPLETED.value
            break

    return plan


def get_next_pending_task(plan: Optional[HierarchicalPlan]) -> Tuple[Optional[str], Optional[str]]:
    """Get the next pending task to execute."""
    if not plan:
        return None, None

    for main_task in plan["main_tasks"]:
        for sub_task in main_task["sub_tasks"]:
            if sub_task["status"] == TaskStatus.PENDING.value:
                return main_task["id"], sub_task["id"]

    return None, None


def process_tool_results(state: ResearchState) -> dict:
    """Process tool results, track findings per task, check for replanning needs, and advance to next task."""
    messages = state.get("messages", [])
    plan = state.get("hierarchical_plan")
    current_main_id = state.get("current_main_task_id")
    current_sub_id = state.get("current_sub_task_id")

    new_findings = []
    new_urls = list(state.get("read_urls", []))
    search_results_content = ""

    for msg in messages[-5:]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content

            if len(content) > 200:
                summary = content[:500] + "..." if len(content) > 500 else content
                if summary not in state.get("findings", []):
                    new_findings.append(summary)
                search_results_content += content + "\n"

    updated_plan = plan
    if plan and current_main_id and current_sub_id and new_findings:
        updated_plan = update_plan_findings(plan, current_main_id, current_sub_id, new_findings)

    needs_replanning = False
    pending_replan_request = None
    replan_count = state.get("replan_count", 0)

    current_query = None
    if plan and current_main_id and current_sub_id:
        from .research import find_current_task
        current_query, _ = find_current_task(plan, current_main_id, current_sub_id)

    if (search_results_content and
        replan_count < MAX_REPLANS and
        current_sub_id and current_sub_id.endswith(".1")):

        from langchain_openai import ChatOpenAI
        from .replanning import analyze_for_replanning
        llm = ChatOpenAI()
        replan_request = analyze_for_replanning(
            state["query"],
            current_query or "",
            search_results_content,
            llm
        )

        if replan_request:
            needs_replanning = True
            pending_replan_request = replan_request

    next_main_id, next_sub_id = get_next_pending_task(updated_plan)

    return {
        "findings": state.get("findings", []) + new_findings,
        "read_urls": new_urls,
        "hierarchical_plan": updated_plan,
        "current_main_task_id": next_main_id,
        "current_sub_task_id": next_sub_id,
        "needs_replanning": needs_replanning,
        "pending_replan_request": pending_replan_request,
    }
