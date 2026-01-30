"""Planning module for research agent.

Contains planning-related logic:
- Intent analysis
- Task decomposition
- Plan validation
- Plan refinement
"""
import json
import re
import copy
from typing import Dict, Any, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import HierarchicalPlan, MainTask, SubTask, TaskStatus
from ..prompts import (
    INTENT_ANALYZER_PROMPT,
    TASK_DECOMPOSER_PROMPT,
    PLAN_VALIDATOR_PROMPT,
)


def parse_json_response(content: str) -> Optional[Dict]:
    """Safely parse JSON from LLM response. Returns dict or None."""
    def try_parse(s: str) -> Optional[Dict]:
        """Try multiple parsing strategies. Returns dict or None."""
        if not s or not isinstance(s, str):
            return None

        strategies = [
            lambda x: json.loads(x),
            lambda x: json.loads(re.sub(r'\s+', ' ', x)),
            lambda x: json.loads(x.strip()),
        ]

        for strategy in strategies:
            try:
                result = strategy(s)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return None

    if '```json' in content:
        try:
            start = content.find('```json') + 7
            end = content.find('```', start)
            if end > start:
                json_str = content[start:end].strip()
                result = try_parse(json_str)
                if result:
                    return result
        except Exception:
            pass

    if '```' in content:
        try:
            start = content.find('```') + 3
            newline = content.find('\n', start)
            if newline != -1 and newline - start < 15:
                start = newline + 1
            end = content.find('```', start)
            if end > start:
                json_str = content[start:end].strip()
                if json_str.startswith('{'):
                    result = try_parse(json_str)
                    if result:
                        return result
        except Exception:
            pass

    try:
        start = content.find('{')
        if start != -1:
            depth = 0
            end = start
            for i, char in enumerate(content[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end > start:
                json_str = content[start:end]
                result = try_parse(json_str)
                if result:
                    return result
    except Exception:
        pass

    return None


def analyze_intent(query: str, llm: ChatOpenAI) -> Dict[str, Any]:
    """Phase 1: Analyze query to identify independent intents."""
    messages = [
        SystemMessage(content=INTENT_ANALYZER_PROMPT),
        HumanMessage(content=f"Analyze this research query: {query}")
    ]

    fallback = {
        "analysis": "Single topic query",
        "is_sequential": False,
        "intent_count": 1,
        "main_topics": [{"topic": query[:30], "description": query}]
    }

    try:
        response = llm.invoke(messages)
        result = parse_json_response(str(response.content))

        if not isinstance(result, dict):
            return fallback

        main_topics = result.get("main_topics")
        if not isinstance(main_topics, list) or not main_topics:
            return fallback

        valid_topics = []
        for topic in main_topics:
            if isinstance(topic, dict) and topic.get("topic"):
                valid_topics.append({
                    "topic": str(topic.get("topic", ""))[:30],
                    "description": str(topic.get("description", query))
                })

        if not valid_topics:
            return fallback

        return {
            "analysis": str(result.get("analysis", ""))[:100],
            "is_sequential": bool(result.get("is_sequential", False)),
            "intent_count": len(valid_topics),
            "main_topics": valid_topics
        }

    except Exception:
        return fallback


def decompose_task(
    topic: str,
    description: str,
    original_query: str,
    llm: ChatOpenAI
) -> list[Dict[str, str]]:
    """Phase 2: Decompose a main task into sub-tasks."""
    prompt = TASK_DECOMPOSER_PROMPT.format(
        topic=topic,
        description=description,
        original_query=original_query
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Generate specific search queries for this topic.")
    ]

    fallback = [{"query": f"{topic} {description[:50]}", "purpose": description}]

    try:
        response = llm.invoke(messages)
        result = parse_json_response(str(response.content))

        if not isinstance(result, dict):
            return fallback

        sub_tasks = result.get("sub_tasks")
        if not isinstance(sub_tasks, list) or not sub_tasks:
            return fallback

        valid_tasks = []
        for st in sub_tasks:
            if isinstance(st, dict) and st.get("query"):
                valid_tasks.append({
                    "query": str(st.get("query", "")),
                    "purpose": str(st.get("purpose", ""))
                })

        return valid_tasks if valid_tasks else fallback

    except Exception:
        return fallback


def validate_plan(
    original_query: str,
    plan: HierarchicalPlan,
    llm: ChatOpenAI
) -> Dict[str, Any]:
    """Phase 3: Validate the plan is comprehensive."""
    default_result = {"is_valid": True, "confidence": 0.7, "issues": [], "suggestions": []}

    try:
        plan_lines = []
        for mt in plan["main_tasks"]:
            plan_lines.append(f"Main Task: {mt['topic']}")
            plan_lines.append(f"  Description: {mt['description']}")
            for st in mt["sub_tasks"]:
                plan_lines.append(f"  - Sub Task: {st['query']}")

        plan_summary = "\n".join(plan_lines)

        prompt = PLAN_VALIDATOR_PROMPT.format(
            original_query=original_query,
            plan_summary=plan_summary
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Validate this research plan.")
        ]

        response = llm.invoke(messages)
        result = parse_json_response(str(response.content))

        if not isinstance(result, dict):
            return default_result

        return {
            "is_valid": bool(result.get("is_valid", True)),
            "confidence": float(result.get("confidence", 0.7)),
            "issues": list(result.get("issues", [])),
            "suggestions": list(result.get("suggestions", []))
        }

    except Exception:
        return default_result


def refine_plan(
    plan: HierarchicalPlan,
    validation_result: Dict,
    llm: ChatOpenAI
) -> HierarchicalPlan:
    """Phase 4: Refine plan based on validation feedback."""
    plan = copy.deepcopy(plan)

    suggestions = validation_result.get("suggestions", [])
    if not suggestions or not plan["main_tasks"]:
        return plan

    for suggestion in suggestions[:2]:
        if isinstance(suggestion, str) and suggestion.strip():
            new_sub_task: SubTask = {
                "id": f"{plan['main_tasks'][0]['id']}.{len(plan['main_tasks'][0]['sub_tasks']) + 1}",
                "query": suggestion,
                "status": TaskStatus.PENDING.value,
                "findings": []
            }
            plan["main_tasks"][0]["sub_tasks"].append(new_sub_task)

    plan["refinement_count"] += 1
    return plan


def build_hierarchical_plan(query: str, llm: ChatOpenAI) -> HierarchicalPlan:
    """Build hierarchical plan through planning pipeline.

    Pipeline phases:
    1. Intent Analysis - Identify main topics
    2. Task Decomposition - Create sub-tasks for each main topic
    3. Validation - Check if plan is comprehensive
    4. Refinement (if needed) - Improve the plan
    """
    def create_fallback_plan() -> HierarchicalPlan:
        return {
            "original_query": query,
            "intent_count": 1,
            "main_tasks": [{
                "id": "1",
                "topic": query[:30],
                "description": query,
                "status": TaskStatus.PENDING.value,
                "sub_tasks": [{
                    "id": "1.1",
                    "query": query,
                    "status": TaskStatus.PENDING.value,
                    "findings": []
                }],
                "summary": ""
            }],
            "is_validated": True,
            "refinement_count": 0
        }

    try:
        intent_result = analyze_intent(query, llm)

        main_tasks: list[MainTask] = []

        main_topics = intent_result.get("main_topics", [])
        if not main_topics:
            return create_fallback_plan()

        for idx, topic_info in enumerate(main_topics, start=1):
            topic = str(topic_info.get("topic", f"Topic {idx}"))[:50]
            description = str(topic_info.get("description", query))

            sub_task_infos = decompose_task(topic, description, query, llm)

            sub_tasks: list[SubTask] = []
            for sub_idx, st_info in enumerate(sub_task_infos, start=1):
                sub_task: SubTask = {
                    "id": f"{idx}.{sub_idx}",
                    "query": str(st_info.get("query", ""))[:200],
                    "status": TaskStatus.PENDING.value,
                    "findings": []
                }
                sub_tasks.append(sub_task)

            if not sub_tasks:
                sub_tasks = [{
                    "id": f"{idx}.1",
                    "query": f"{topic} {description[:50]}",
                    "status": TaskStatus.PENDING.value,
                    "findings": []
                }]

            main_task: MainTask = {
                "id": str(idx),
                "topic": topic,
                "description": description,
                "status": TaskStatus.PENDING.value,
                "sub_tasks": sub_tasks,
                "summary": ""
            }
            main_tasks.append(main_task)

        if not main_tasks:
            return create_fallback_plan()

        plan: HierarchicalPlan = {
            "original_query": query,
            "intent_count": intent_result.get("intent_count", 1),
            "main_tasks": main_tasks,
            "is_validated": False,
            "refinement_count": 0
        }

        validation_result = validate_plan(query, plan, llm)

        if not validation_result.get("is_valid", True) and plan["refinement_count"] < 1:
            plan = refine_plan(plan, validation_result, llm)

        plan["is_validated"] = True
        return plan

    except Exception:
        return create_fallback_plan()


def get_first_pending_task(plan: HierarchicalPlan) -> Tuple[Optional[str], Optional[str]]:
    """Get the first pending sub-task from the plan."""
    for main_task in plan["main_tasks"]:
        for sub_task in main_task["sub_tasks"]:
            if sub_task["status"] == TaskStatus.PENDING.value:
                return main_task["id"], sub_task["id"]
    return None, None
