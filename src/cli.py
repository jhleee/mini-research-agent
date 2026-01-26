"""CLI interface for the Deep Research Agent."""
import asyncio
import sys
import os

# Fix Windows encoding
if sys.platform == "win32":
    os.system("")  # Enable ANSI escape sequences
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table

from .agent import run_research


console = Console(force_terminal=True)


def print_banner():
    """Print the application banner."""
    banner = """
+---------------------------------------------------------------+
|                   Deep Research Agent                         |
|                                                               |
|  Powered by GLM-4.7 + LangGraph + Z.AI MCP Tools              |
|  Enter your research question and get a comprehensive report  |
+---------------------------------------------------------------+
"""
    console.print(banner, style="bold cyan")


def progress_callback(node_name: str, state: dict):
    """Callback to display progress updates."""
    status_map = {
        "plan": "[Plan] Creating research plan...",
        "research": "[Search] Researching...",
        "tools": "[Tools] Executing tools...",
        "process_results": "[Process] Processing results...",
        "synthesize": "[Write] Synthesizing report...",
    }

    status = status_map.get(node_name, f"[...] {node_name}...")

    if isinstance(state, dict):
        iteration = state.get("iteration", 0)
        findings_count = len(state.get("findings", []))
        queries_done = len(state.get("search_queries", []))
        plan_count = len(state.get("research_plan", []))

        if plan_count > 0:
            status += f" (Query {queries_done}/{plan_count}, Findings: {findings_count})"


async def research_with_progress(query: str) -> str:
    """Run research with progress display."""
    console.print("\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting research...", total=None)

        current_status = {"stage": "Starting"}

        def update_progress(node_name: str, state: dict):
            nonlocal current_status

            status_map = {
                "plan": "[Plan] Creating research plan...",
                "research": "[Search] Executing search queries...",
                "tools": "[Tools] Fetching web content...",
                "process_results": "[Process] Analyzing results...",
                "synthesize": "[Write] Writing final report...",
            }

            desc = status_map.get(node_name, "[...] Processing...")

            if isinstance(state, dict):
                iteration = state.get("iteration", 0)
                findings = len(state.get("findings", []))
                queries = len(state.get("search_queries", []))
                total = len(state.get("research_plan", []))

                if total > 0:
                    desc += f" [{queries}/{total} queries, {findings} findings]"

            progress.update(task, description=desc)
            current_status["stage"] = node_name

        result = await run_research(query, callback=update_progress)

        progress.update(task, description="[Done] Research complete!")

    return result


async def interactive_mode():
    """Run the agent in interactive mode."""
    print_banner()

    console.print("[dim]Type 'quit' or 'exit' to stop[/dim]\n")

    while True:
        try:
            console.print("[bold green]Research Question:[/bold green]")
            query = console.input("[bold blue]> [/bold blue]").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                console.print("\n[yellow]Goodbye! Happy researching![/yellow]")
                break

            # Run research
            report = await research_with_progress(query)

            # Display report
            console.print("\n")
            console.print(Panel(
                Markdown(report),
                title="[bold green]Research Report[/bold green]",
                border_style="green",
                padding=(1, 2),
            ))
            console.print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Research interrupted. Type 'quit' to exit.[/yellow]\n")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


async def single_query_mode(query: str):
    """Run a single research query."""
    print_banner()

    report = await research_with_progress(query)

    # Display report
    console.print("\n")
    console.print(Panel(
        Markdown(report),
        title="[bold green]Research Report[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        asyncio.run(single_query_mode(query))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
