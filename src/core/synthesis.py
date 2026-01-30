"""Synthesis module for research agent.

Contains report synthesis logic:
- Synthesize findings into comprehensive report
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ..state import ResearchState
from ..prompts import SYNTHESIZER_PROMPT, HIERARCHICAL_SYNTHESIZER_PROMPT


def synthesize_node(state: ResearchState) -> dict:
    """Synthesize findings into hierarchical report organized by main tasks."""
    llm = ChatOpenAI()
    plan = state.get("hierarchical_plan")

    if plan and plan["main_tasks"]:
        topics_summary = []
        all_findings = []

        for main_task in plan["main_tasks"]:
            topic_findings = []
            for sub_task in main_task["sub_tasks"]:
                if sub_task["findings"]:
                    topic_findings.extend(sub_task["findings"])

            topics_summary.append(f"- {main_task['topic']}: {main_task['description']}")

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
