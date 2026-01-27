"""Deep Research Agent using LangGraph."""
import json
import asyncio
import copy
from typing import Literal, Dict, Any, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, MAX_ITERATIONS
from .state import ResearchState, HierarchicalPlan, MainTask, SubTask, TaskStatus, DynamicQueryRequest
from .tools import TOOLS
from .prompts import (
    PLANNER_PROMPT,
    RESEARCHER_PROMPT,
    SYNTHESIZER_PROMPT,
    INTENT_ANALYZER_PROMPT,
    TASK_DECOMPOSER_PROMPT,
    PLAN_VALIDATOR_PROMPT,
    HIERARCHICAL_SYNTHESIZER_PROMPT,
    REPLAN_ANALYZER_PROMPT,
    DYNAMIC_QUERY_GENERATOR_PROMPT,
)
from .config import MAX_ITERATIONS, MAX_REPLANS


def create_llm():
    """Create the GLM LLM client."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL,
        temperature=0.7,
    )


def parse_json_response(content: str) -> Optional[Dict]:
    """Safely parse JSON from LLM response. Always returns a dict or None."""
    import re

    if not content or not isinstance(content, str):
        return None

    def try_parse(s: str) -> Optional[Dict]:
        """Try multiple parsing strategies. Returns dict or None."""
        if not s or not isinstance(s, str):
            return None

        strategies = [
            # Strategy 1: Direct parse
            lambda x: json.loads(x),
            # Strategy 2: Normalize whitespace
            lambda x: json.loads(re.sub(r'\s+', ' ', x)),
            # Strategy 3: Strip and parse
            lambda x: json.loads(x.strip()),
        ]

        for strategy in strategies:
            try:
                result = strategy(s)
                # Only return if result is a dict
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        return None

    # Try to find JSON in ```json blocks first
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

    # Try to find JSON in generic ``` blocks
    if '```' in content:
        try:
            start = content.find('```') + 3
            # Skip language identifier if present
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

    # Try to find JSON object in raw response - find matching braces
    try:
        start = content.find('{')
        if start != -1:
            # Find the matching closing brace
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
        "intent_count": 1,
        "main_topics": [{"topic": query[:30], "description": query}]
    }

    try:
        response = llm.invoke(messages)
        result = parse_json_response(response.content)

        # Ensure result is a dict
        if not isinstance(result, dict):
            return fallback

        # Validate main_topics is a list
        main_topics = result.get("main_topics")
        if not isinstance(main_topics, list) or not main_topics:
            return fallback

        # Validate each topic has required fields
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
        result = parse_json_response(response.content)

        # Ensure result is a dict and has sub_tasks key
        if not isinstance(result, dict):
            return fallback

        sub_tasks = result.get("sub_tasks")
        if not isinstance(sub_tasks, list) or not sub_tasks:
            return fallback

        # Validate each sub_task has required fields
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
        # Build plan summary
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
        result = parse_json_response(response.content)

        # Ensure result is a dict with expected structure
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

    # Add suggested queries as additional sub-tasks to the most relevant main task
    for suggestion in suggestions[:2]:  # Limit refinements
        if isinstance(suggestion, str) and suggestion.strip():
            # Add to first main task (simplified logic)
            new_sub_task: SubTask = {
                "id": f"{plan['main_tasks'][0]['id']}.{len(plan['main_tasks'][0]['sub_tasks']) + 1}",
                "query": suggestion,
                "status": TaskStatus.PENDING.value,
                "findings": []
            }
            plan["main_tasks"][0]["sub_tasks"].append(new_sub_task)

    plan["refinement_count"] += 1
    return plan


def analyze_for_replanning(
    original_query: str,
    current_query: str,
    search_results: str,
    llm: ChatOpenAI
) -> Optional[DynamicQueryRequest]:
    """Analyze search results to determine if dynamic replanning is needed.

    Returns a DynamicQueryRequest if replanning is needed, None otherwise.
    """
    prompt = REPLAN_ANALYZER_PROMPT.format(
        original_query=original_query,
        current_query=current_query,
        search_results=search_results[:3000]  # Limit context size
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Analyze these search results for dynamic replanning needs.")
    ]

    try:
        response = llm.invoke(messages)
        result = parse_json_response(response.content)

        if not isinstance(result, dict):
            return None

        if not result.get("needs_replanning", False):
            return None

        extracted_items = result.get("extracted_items", [])
        if not extracted_items or not isinstance(extracted_items, list):
            return None

        # Limit items to MAX_DYNAMIC_ITEMS
        from .config import MAX_DYNAMIC_ITEMS
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
) -> list[Dict[str, str]]:
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
        result = parse_json_response(response.content)

        if not isinstance(result, dict):
            # Fallback: generate queries from template
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
            # Fallback
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
        # Fallback: generate queries from template
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
    dynamic_queries: list[Dict[str, str]],
    parent_main_task_id: str
) -> HierarchicalPlan:
    """Add dynamically generated queries as new sub-tasks to the plan."""
    plan = copy.deepcopy(plan)

    # Find the parent main task
    for main_task in plan["main_tasks"]:
        if main_task["id"] == parent_main_task_id:
            # Get the next sub-task ID
            existing_count = len(main_task["sub_tasks"])

            for idx, query_info in enumerate(dynamic_queries, start=1):
                new_sub_task: SubTask = {
                    "id": f"{parent_main_task_id}.{existing_count + idx}",
                    "query": str(query_info.get("query", ""))[:200],
                    "status": TaskStatus.PENDING.value,
                    "findings": []
                }
                main_task["sub_tasks"].append(new_sub_task)

            # Reset main task status to in_progress if it was completed
            if main_task["status"] == TaskStatus.COMPLETED.value:
                main_task["status"] = TaskStatus.IN_PROGRESS.value

            break

    return plan


def build_hierarchical_plan(query: str, llm: ChatOpenAI) -> HierarchicalPlan:
    """Build a hierarchical plan through the agentic planning pipeline.

    Pipeline phases:
    1. Intent Analysis - Identify main topics
    2. Task Decomposition - Create sub-tasks for each main topic
    3. Validation - Check if plan is comprehensive
    4. Refinement (if needed) - Improve the plan
    """
    # Fallback plan in case anything goes wrong
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
        # Phase 1: Intent Analysis
        intent_result = analyze_intent(query, llm)

        # Phase 2: Build main tasks and decompose each
        main_tasks: list[MainTask] = []

        main_topics = intent_result.get("main_topics", [])
        if not main_topics:
            return create_fallback_plan()

        for idx, topic_info in enumerate(main_topics, start=1):
            topic = str(topic_info.get("topic", f"Topic {idx}"))[:50]
            description = str(topic_info.get("description", query))

            # Decompose into sub-tasks
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

            # Ensure at least one sub-task
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

        # Ensure at least one main task
        if not main_tasks:
            return create_fallback_plan()

        # Build initial plan
        plan: HierarchicalPlan = {
            "original_query": query,
            "intent_count": intent_result.get("intent_count", 1),
            "main_tasks": main_tasks,
            "is_validated": False,
            "refinement_count": 0
        }

        # Phase 3: Validation
        validation_result = validate_plan(query, plan, llm)

        # Phase 4: Refinement if needed (max 1 refinement to avoid loops)
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


def plan_node(state: ResearchState) -> dict:
    """Create a hierarchical research plan from the query using agentic planning.

    This uses a multi-phase planning pipeline:
    1. Intent Analysis - Identify independent main topics
    2. Task Decomposition - Create sub-tasks for each main topic
    3. Validation - Check if plan is comprehensive
    4. Refinement - Improve plan if needed
    """
    llm = create_llm()
    query = state["query"]

    # Build hierarchical plan through the agentic pipeline
    plan = build_hierarchical_plan(query, llm)

    # Convert to flat list for backward compatibility
    flat_plan = []
    for main_task in plan["main_tasks"]:
        for sub_task in main_task["sub_tasks"]:
            flat_plan.append(sub_task["query"])

    # Get first pending task
    first_main_id, first_sub_id = get_first_pending_task(plan)

    # Build summary message
    task_summary = []
    for mt in plan["main_tasks"]:
        task_summary.append(f"[{mt['topic']}]: {len(mt['sub_tasks'])} queries")

    return {
        "hierarchical_plan": plan,
        "research_plan": flat_plan,  # Backward compatibility
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


def research_node(state: ResearchState) -> dict:
    """Execute research using tools, with hierarchical task awareness."""
    llm = create_llm().bind_tools(TOOLS)

    # Get current task context from hierarchical plan
    plan = state.get("hierarchical_plan")
    current_main_id = state.get("current_main_task_id")
    current_sub_id = state.get("current_sub_task_id")

    # Find the current sub-task to execute
    current_query = None
    current_main_topic = None

    if plan and current_main_id and current_sub_id:
        current_query, current_main_topic = find_current_task(plan, current_main_id, current_sub_id)

    # Fallback to flat plan if hierarchical lookup fails
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

    # Build context header with main task awareness
    context_header = ""
    if current_main_topic:
        context_header = f"Main Research Topic: {current_main_topic}\n"

    # Build context from previous findings
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

    # Add previous messages for context
    messages.extend(state.get("messages", [])[-10:])

    response = llm.invoke(messages)
    executed = state.get("search_queries", [])

    return {
        "messages": [response],
        "search_queries": executed + [current_query],
        "iteration": state.get("iteration", 0) + 1,
    }


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
            # Update main task status to in_progress if it was pending
            if main_task["status"] == TaskStatus.PENDING.value:
                main_task["status"] = TaskStatus.IN_PROGRESS.value
            # Check if all sub-tasks are complete
            if all(st["status"] == TaskStatus.COMPLETED.value for st in main_task["sub_tasks"]):
                main_task["status"] = TaskStatus.COMPLETED.value
            break

    return plan


def get_next_pending_task(plan: Optional[HierarchicalPlan]) -> Tuple[Optional[str], Optional[str]]:
    """Get the next pending task to execute."""
    if not plan:
        return None, None

    # Find next pending sub-task
    for main_task in plan["main_tasks"]:
        for sub_task in main_task["sub_tasks"]:
            if sub_task["status"] == TaskStatus.PENDING.value:
                return main_task["id"], sub_task["id"]

    return None, None  # All tasks complete


def replan_node(state: ResearchState) -> dict:
    """Process dynamic replanning request and add new sub-tasks to the plan.

    This node is called when search results indicate that follow-up queries
    are needed based on discovered items (e.g., "find 3 pasta types" → search each).
    """
    llm = create_llm()
    replan_request = state.get("pending_replan_request")
    plan = state.get("hierarchical_plan")
    current_main_id = state.get("current_main_task_id")

    if not replan_request or not plan:
        return {
            "needs_replanning": False,
            "pending_replan_request": None,
        }

    # Use the current main task ID or default to "1"
    target_main_id = current_main_id or "1"

    # Generate dynamic queries for extracted items
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

    # Add dynamic queries as new sub-tasks
    updated_plan = add_dynamic_tasks_to_plan(
        plan,
        dynamic_queries,
        target_main_id
    )

    # Update flat plan for backward compatibility
    flat_plan = state.get("research_plan", [])[:]
    for query_info in dynamic_queries:
        query = query_info.get("query", "")
        if query and query not in flat_plan:
            flat_plan.append(query)

    # Get the first new pending task
    next_main_id, next_sub_id = get_next_pending_task(updated_plan)

    return {
        "hierarchical_plan": updated_plan,
        "research_plan": flat_plan,
        "current_main_task_id": next_main_id,
        "current_sub_task_id": next_sub_id,
        "needs_replanning": False,
        "pending_replan_request": None,
        "replan_count": state.get("replan_count", 0) + 1,
        "messages": [AIMessage(content=f"Dynamic replanning: Added {len(dynamic_queries)} follow-up queries for items: {', '.join(replan_request['extracted_items'])}")],
    }


def process_tool_results(state: ResearchState) -> dict:
    """Process tool results, track findings per task, check for replanning needs, and advance to next task."""
    messages = state.get("messages", [])
    plan = state.get("hierarchical_plan")
    current_main_id = state.get("current_main_task_id")
    current_sub_id = state.get("current_sub_task_id")

    # Look for tool results in recent messages
    new_findings = []
    new_urls = list(state.get("read_urls", []))
    search_results_content = ""

    for msg in messages[-5:]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content

            # Check if this looks like substantial content
            if len(content) > 200:
                # Extract a summary finding
                summary = content[:500] + "..." if len(content) > 500 else content
                if summary not in state.get("findings", []):
                    new_findings.append(summary)
                # Accumulate for replanning analysis
                search_results_content += content + "\n"

    # Update hierarchical plan with findings if available
    updated_plan = plan
    if plan and current_main_id and current_sub_id and new_findings:
        updated_plan = update_plan_findings(plan, current_main_id, current_sub_id, new_findings)

    # Check if dynamic replanning is needed (only if we haven't exceeded replan limit)
    needs_replanning = False
    pending_replan_request = None
    replan_count = state.get("replan_count", 0)

    # Find current query for replanning analysis
    current_query = None
    if plan and current_main_id and current_sub_id:
        current_query, _ = find_current_task(plan, current_main_id, current_sub_id)

    # Only attempt replanning if:
    # 1. We have search results
    # 2. We haven't exceeded the replan limit
    # 3. We're on the first sub-task (typically the discovery phase)
    if (search_results_content and
        replan_count < MAX_REPLANS and
        current_sub_id and current_sub_id.endswith(".1")):  # First sub-task of a main task

        llm = create_llm()
        replan_request = analyze_for_replanning(
            state["query"],
            current_query or "",
            search_results_content,
            llm
        )

        if replan_request:
            needs_replanning = True
            pending_replan_request = replan_request

    # Advance to next sub-task
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


def synthesize_node(state: ResearchState) -> dict:
    """Synthesize findings into a hierarchical report organized by main tasks."""
    llm = create_llm()
    plan = state.get("hierarchical_plan")

    if plan and plan["main_tasks"]:
        # Build hierarchical synthesis with findings per main task
        topics_summary = []
        all_findings = []

        for main_task in plan["main_tasks"]:
            # Collect findings from all sub-tasks of this main task
            topic_findings = []
            for sub_task in main_task["sub_tasks"]:
                if sub_task["findings"]:
                    topic_findings.extend(sub_task["findings"])

            topics_summary.append(f"- {main_task['topic']}: {main_task['description']}")

            # Build section for this main task
            if topic_findings:
                section = f"## {main_task['topic']}\n\n" + "\n\n".join(topic_findings)
            else:
                section = f"## {main_task['topic']}\n\nNo specific findings for this topic."
            all_findings.append(section)

        system_prompt = HIERARCHICAL_SYNTHESIZER_PROMPT.format(
            intent_count=plan["intent_count"],
            topics_summary="\n".join(topics_summary)
        )

        findings_text = "\n\n---\n\n".join(all_findings) if all_findings else "No findings."
    else:
        # Fallback to flat synthesis
        system_prompt = SYNTHESIZER_PROMPT
        findings = state.get("findings", ["No specific findings were gathered."])
        findings_text = "\n\n---\n\n".join(findings)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Original Research Query: {state['query']}

Research Findings:
{findings_text}

Please synthesize these findings into a comprehensive research report.""")
    ]

    response = llm.invoke(messages)

    return {
        "report": response.content,
        "status": "done",
        "messages": [AIMessage(content="Research synthesis complete.")]
    }


def should_continue(state: ResearchState) -> Literal["research", "tools", "synthesize", "end"]:
    """Determine the next step in the workflow."""
    status = state.get("status", "planning")

    if status == "done":
        return "end"

    if status == "synthesizing":
        return "synthesize"

    # Check iteration limit
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "synthesize"

    # Check if there are tool calls to process
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

    return "research"


def should_continue_after_process(state: ResearchState) -> Literal["research", "replan"]:
    """Determine whether to continue research or do dynamic replanning."""
    # Check if dynamic replanning is needed
    if state.get("needs_replanning", False) and state.get("pending_replan_request"):
        return "replan"

    return "research"


def create_research_graph():
    """Create the research agent graph with dynamic replanning support."""
    # Create the graph
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("process_results", process_tool_results)
    graph.add_node("replan", replan_node)  # New: dynamic replanning node
    graph.add_node("synthesize", synthesize_node)

    # Set entry point
    graph.set_entry_point("plan")

    # Add edges
    graph.add_edge("plan", "research")

    # Conditional routing after research
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

    # After tools, process results
    graph.add_edge("tools", "process_results")

    # After processing results, decide: replan or continue research
    graph.add_conditional_edges(
        "process_results",
        should_continue_after_process,
        {
            "research": "research",
            "replan": "replan",
        }
    )

    # After replanning, continue with research
    graph.add_edge("replan", "research")

    # Synthesize leads to end
    graph.add_edge("synthesize", END)

    return graph.compile()


async def run_research_with_tools(query: str, callback=None, config={ "recursion_limit": 200}) -> str:
    """Run research with detailed event callbacks for tool usage.

    Args:
        query: The research question.
        callback: Callback function(event_type, data) for events.

    Returns:
        The final research report.
    """
    graph = create_research_graph()

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
        # Dynamic replanning fields
        "needs_replanning": False,
        "pending_replan_request": None,
        "replan_count": 0,
    }

    def emit(event_type: str, data: dict):
        if callback:
            callback(event_type, data)

    final_state = None
    plan_emitted = False  # Track if plan_created event has been emitted
    last_main_task_id = None
    last_sub_task_id = None

    async for state in graph.astream(initial_state, config=config):
        final_state = state

        for node_name, node_state in state.items():
            # Emit node start event
            emit("node_start", {"node": node_name})

            # Emit LLM loading event for LLM-intensive nodes
            llm_nodes = {
                "plan": "Planning research",
                "research": "Analyzing",
                "replan": "Dynamic replanning",
                "synthesize": "Writing report",
            }
            if node_name in llm_nodes:
                emit("llm_start", {"message": llm_nodes[node_name]})

            if isinstance(node_state, dict):
                # Emit hierarchical plan when created (only once)
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
                        # Also emit flat plan for backward compatibility
                        flat_queries = []
                        for mt in h_plan["main_tasks"]:
                            for st in mt["sub_tasks"]:
                                flat_queries.append(st["query"])
                        emit("plan_created", {"queries": flat_queries})
                        plan_emitted = True

                # Emit dynamic replanning event when plan is updated
                if node_name == "replan":
                    replan_request = node_state.get("pending_replan_request")
                    h_plan = node_state.get("hierarchical_plan")
                    if h_plan:
                        # Emit updated plan with dynamic tasks
                        emit("dynamic_replan", {
                            "replan_count": node_state.get("replan_count", 0),
                            "extracted_items": replan_request.get("extracted_items", []) if replan_request else [],
                            "new_queries": [
                                st["query"]
                                for mt in h_plan["main_tasks"]
                                for st in mt["sub_tasks"]
                                if st["status"] == TaskStatus.PENDING.value
                            ]
                        })

                # Emit task progress updates when task changes
                current_main = node_state.get("current_main_task_id")
                current_sub = node_state.get("current_sub_task_id")

                # Check if we completed a sub-task (task ID changed)
                if last_sub_task_id and (current_sub != last_sub_task_id):
                    emit("task_progress", {
                        "main_task_id": last_main_task_id,
                        "sub_task_id": last_sub_task_id,
                        "status": "completed"
                    })

                # Emit in_progress for new task
                if current_main and current_sub and (current_sub != last_sub_task_id):
                    emit("task_progress", {
                        "main_task_id": current_main,
                        "sub_task_id": current_sub,
                        "status": "in_progress"
                    })
                    last_main_task_id = current_main
                    last_sub_task_id = current_sub

                # Emit search query completion (backward compatibility)
                search_queries = node_state.get("search_queries", [])
                if search_queries:
                    emit("query_completed", {"completed": search_queries})

                # Check for tool calls in messages
                messages = node_state.get("messages", [])
                for msg in messages:
                    # Tool call request
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            emit("tool_call", {
                                "tool": tool_call.get("name", ""),
                                "args": tool_call.get("args", {}),
                            })

                    # Tool result
                    if hasattr(msg, "type") and msg.type == "tool":
                        emit("tool_result", {
                            "tool": getattr(msg, "name", ""),
                            "result": msg.content if hasattr(msg, "content") else "",
                        })

    # Extract final report
    if final_state:
        for node_state in final_state.values():
            if isinstance(node_state, dict) and node_state.get("report"):
                return node_state["report"]

    return "Research could not be completed."
