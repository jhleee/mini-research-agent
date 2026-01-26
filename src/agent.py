"""Deep Research Agent using LangGraph."""
import json
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, MAX_ITERATIONS
from .state import ResearchState
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


def plan_node(state: ResearchState) -> dict:
    """Create a research plan from the query."""
    llm = create_llm()

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Create a research plan for: {state['query']}")
    ]

    response = llm.invoke(messages)
    content = response.content

    # Parse the plan from JSON
    try:
        # Find JSON array in response
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            plan = json.loads(content[start:end])
        else:
            # Fallback: use the query itself
            plan = [state['query']]
    except json.JSONDecodeError:
        plan = [state['query']]

    return {
        "research_plan": plan,
        "search_queries": [],
        "read_urls": [],
        "findings": [],
        "iteration": 0,
        "status": "researching",
        "messages": [AIMessage(content=f"Research plan created with {len(plan)} queries: {plan}")]
    }


def research_node(state: ResearchState) -> dict:
    """Execute research using tools."""
    llm = create_llm().bind_tools(TOOLS)

    # Determine what to research next
    plan = state.get("research_plan", [])
    executed = state.get("search_queries", [])
    remaining = [q for q in plan if q not in executed]

    if not remaining:
        return {
            "status": "synthesizing",
            "messages": [AIMessage(content="All planned searches completed. Moving to synthesis.")]
        }

    current_query = remaining[0]

    # Build context from previous findings
    findings_context = ""
    if state.get("findings"):
        findings_context = f"\n\nPrevious findings:\n" + "\n".join(state["findings"][-3:])

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"""Research query: {state['query']}

Current search focus: {current_query}
{findings_context}

Use the web_search tool to search for: {current_query}
Then read relevant webpages to gather detailed information.""")
    ]

    # Add previous messages for context
    messages.extend(state.get("messages", [])[-10:])

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "search_queries": executed + [current_query],
        "iteration": state.get("iteration", 0) + 1,
    }


def process_tool_results(state: ResearchState) -> dict:
    """Process tool results and extract findings."""
    messages = state.get("messages", [])

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

    return {
        "findings": state.get("findings", []) + new_findings,
        "read_urls": new_urls,
    }


def synthesize_node(state: ResearchState) -> dict:
    """Synthesize findings into a final report."""
    llm = create_llm()

    findings = state.get("findings", [])
    if not findings:
        findings = ["No specific findings were gathered."]

    findings_text = "\n\n---\n\n".join(findings)

    messages = [
        SystemMessage(content=SYNTHESIZER_PROMPT),
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


async def run_research_with_tools(query: str, callback=None) -> str:
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
        "research_plan": [],
        "search_queries": [],
        "read_urls": [],
        "findings": [],
        "iteration": 0,
        "report": "",
        "status": "planning",
    }

    def emit(event_type: str, data: dict):
        if callback:
            callback(event_type, data)

    final_state = None
    plan_emitted = False  # Track if plan_created event has been emitted

    async for state in graph.astream(initial_state):
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
                # Emit research plan when created (only once)
                if not plan_emitted:
                    research_plan = node_state.get("research_plan", [])
                    if research_plan:
                        emit("plan_created", {"queries": research_plan})
                        plan_emitted = True

                # Emit search query completion
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
