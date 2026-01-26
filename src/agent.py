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
from .state import ResearchState, HierarchicalPlan, MainTask, SubTask, TaskStatus
from .tools import TOOLS


def create_llm():
    """Create the GLM LLM client."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model=OPENAI_MODEL,
        temperature=0.7,
    )


# System prompts for different agent roles
PLANNER_PROMPT = """You are a research planner. Your job is to create a comprehensive research plan.

Given a user's research query, break it down into specific sub-questions that need to be answered.
Think step by step about what information is needed to fully answer the query.

Output your plan as a JSON array of search queries, like:
["search query 1", "search query 2", "search query 3"]

Be specific and comprehensive. Include different angles and perspectives.
Aim for 3-5 search queries that together will provide comprehensive coverage of the topic."""

RESEARCHER_PROMPT = """You are a research assistant with access to web search and webpage reading tools.

Your task is to gather information to answer the research query. You have these tools:
- web_search: Search the web for information
- read_webpage: Read the full content of a specific URL

Strategy:
1. Use web_search to find relevant sources
2. Use read_webpage to get detailed content from promising URLs
3. Focus on gathering facts, data, and expert opinions

Be thorough but efficient. Don't read too many pages - focus on the most relevant ones.
After gathering enough information, summarize your key findings."""

SYNTHESIZER_PROMPT = """You are a research synthesizer. Your job is to create a comprehensive report.

Based on the research findings provided, create a well-structured report that:
1. Directly answers the original research query
2. Synthesizes information from multiple sources
3. Presents findings in a clear, organized manner
4. Highlights key insights and conclusions
5. Notes any limitations or areas needing further research

Write in a professional, informative style. Use markdown formatting for clarity."""

# Enhanced Planning Prompts for Hierarchical Task Decomposition
INTENT_ANALYZER_PROMPT = """You are an intent analyzer for a research agent. Your job is to identify independent research topics within a user's query.

Analyze the query and identify:
1. How many distinct, independent research topics are present
2. The main topic/theme for each independent request
3. A brief description of what needs to be researched for each

Rules:
- Topics are INDEPENDENT if they can be researched separately and don't depend on each other's results
- IMPORTANT: When a query lists multiple items connected by "and", "그리고", commas, or conjunctions, each item is likely a SEPARATE independent topic
- Example: "AI trends and quantum computing developments" = 2 independent topics (AI trends, quantum computing)
- Example: "AI 동향과 블록체인 전망" = 2 independent topics (AI 동향, 블록체인 전망)
- Example: "AI, quantum computing, blockchain trends" = 3 independent topics
- Example: "How AI is applied in healthcare diagnosis" = 1 topic (AI application in specific domain)
- Maximum 4 main topics per query
- If query is simple/single-topic, return just 1 main topic
- Each topic name should be SHORT and FOCUSED on a single subject area (e.g., "AI 최신 동향", "양자컴퓨팅 발전", "블록체인 전망")
- Respond in the same language as the user's query

Output ONLY valid JSON in this exact format:
{
    "analysis": "Brief explanation of your analysis",
    "intent_count": <number>,
    "main_topics": [
        {
            "topic": "Short topic name (max 30 chars)",
            "description": "What needs to be researched about this topic"
        }
    ]
}"""

TASK_DECOMPOSER_PROMPT = """You are a research task decomposer. Given a main research topic, break it down into specific search queries.

Main Topic: {topic}
Description: {description}
Original User Query Context: {original_query}

Create 2-4 specific search queries that will comprehensively cover THIS TOPIC ONLY.

CRITICAL RULES:
- Each query must focus ONLY on "{topic}" - do NOT include other topics from the original query
- Keep queries SHORT and FOCUSED (ideally 3-6 words)
- Do NOT combine multiple unrelated subjects in one query
- Each query should cover a DIFFERENT ASPECT of this single topic
- Be in the same language as the original query
- Be suitable for web search engines

BAD examples (mixing topics):
- "AI 동향 양자컴퓨팅 블록체인" (mixing 3 topics)
- "AI trends quantum computing blockchain" (mixing 3 topics)

GOOD examples (focused on single topic):
- For "AI 최신 동향": "AI 최신 동향 2025", "생성형 AI 트렌드", "AI 산업 적용 사례"
- For "양자컴퓨팅": "양자컴퓨팅 최신 발전", "양자컴퓨터 상용화 현황"
- For "블록체인": "블록체인 기술 전망 2025", "블록체인 실제 활용 사례"

Output ONLY valid JSON in this exact format:
{
    "sub_tasks": [
        {
            "query": "The specific search query",
            "purpose": "What this query will help discover"
        }
    ]
}"""

PLAN_VALIDATOR_PROMPT = """You are a research plan validator. Review the following research plan and determine if it's comprehensive enough.

Original Query: {original_query}

Research Plan:
{plan_summary}

Evaluate:
1. Does the plan cover all aspects of the original query?
2. Are there any missing important angles?
3. Are the search queries specific enough?
4. Is there unnecessary overlap between queries?

Output ONLY valid JSON in this exact format:
{
    "is_valid": true,
    "issues": [],
    "suggestions": [],
    "confidence": 0.85
}

Note: Set is_valid to false only if there are critical gaps in the plan."""

HIERARCHICAL_SYNTHESIZER_PROMPT = """You are a research synthesizer creating a comprehensive report.

The research covered {intent_count} main topic(s):
{topics_summary}

Based on the findings for each topic, create a well-structured report that:
1. Has a clear section for each main topic
2. Synthesizes findings within each section
3. Highlights connections between topics if relevant
4. Provides a unified conclusion

Use markdown formatting. Write in a professional, informative style.
Respond in the same language as the original query."""


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


def process_tool_results(state: ResearchState) -> dict:
    """Process tool results, track findings per task, and advance to next task."""
    messages = state.get("messages", [])
    plan = state.get("hierarchical_plan")
    current_main_id = state.get("current_main_task_id")
    current_sub_id = state.get("current_sub_task_id")

    # Look for tool results in recent messages
    new_findings = []
    new_urls = list(state.get("read_urls", []))

    for msg in messages[-5:]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content

            # Check if this looks like substantial content
            if len(content) > 200:
                # Extract a summary finding
                summary = content[:500] + "..." if len(content) > 500 else content
                if summary not in state.get("findings", []):
                    new_findings.append(summary)

    # Update hierarchical plan with findings if available
    updated_plan = plan
    if plan and current_main_id and current_sub_id and new_findings:
        updated_plan = update_plan_findings(plan, current_main_id, current_sub_id, new_findings)

    # Advance to next sub-task
    next_main_id, next_sub_id = get_next_pending_task(updated_plan)

    return {
        "findings": state.get("findings", []) + new_findings,
        "read_urls": new_urls,
        "hierarchical_plan": updated_plan,
        "current_main_task_id": next_main_id,
        "current_sub_task_id": next_sub_id,
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


def create_research_graph():
    """Create the research agent graph."""
    # Create the graph
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("process_results", process_tool_results)
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

    # After tools, process results and continue research
    graph.add_edge("tools", "process_results")
    graph.add_edge("process_results", "research")

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
