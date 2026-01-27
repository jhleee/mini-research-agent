"""Controller for the Research Agent TUI."""
import asyncio
from typing import Callable, Optional

from .models import AppState, MessageRole, TaskStatus, Task
from ..agent import run_research_with_tools

# Debug logging to file
DEBUG_LOG = open("debug.log", "w", encoding="utf-8")

def debug_log(msg: str):
    DEBUG_LOG.write(f"{msg}\n")
    DEBUG_LOG.flush()


class ResearchController:
    """Controller that manages the research agent and state."""

    def __init__(self):
        self.state = AppState()
        self._on_state_change: Optional[Callable] = None
        self._on_message: Optional[Callable] = None
        self._on_task_update: Optional[Callable] = None
        self._on_loading: Optional[Callable] = None
        self._on_tasks_clear: Optional[Callable] = None
        self._on_tool_call: Optional[Callable] = None
        self._on_tool_result: Optional[Callable] = None
        self._research_task: Optional[asyncio.Task] = None
        self._current_tool_widget_id: Optional[str] = None

    def set_callbacks(
        self,
        on_state_change: Optional[Callable] = None,
        on_message: Optional[Callable] = None,
        on_task_update: Optional[Callable] = None,
        on_loading: Optional[Callable] = None,
        on_tasks_clear: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
        on_tool_result: Optional[Callable] = None,
    ):
        self._on_state_change = on_state_change
        self._on_message = on_message
        self._on_task_update = on_task_update
        self._on_loading = on_loading
        self._on_tasks_clear = on_tasks_clear
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result

    def _notify_state_change(self):
        if self._on_state_change:
            self._on_state_change(self.state)

    def _notify_message(self, role: MessageRole, content: str):
        msg = self.state.add_message(role, content)
        if self._on_message:
            self._on_message(msg)
        self._notify_state_change()

    def _notify_task_update(self, task: Task):
        if self._on_task_update:
            self._on_task_update(task)
        self._notify_state_change()

    def _show_loading(self, text: str = "LLM responding"):
        if self._on_loading:
            self._on_loading(True, text)

    def _hide_loading(self):
        if self._on_loading:
            self._on_loading(False, "")

    def add_user_message(self, content: str):
        self._notify_message(MessageRole.USER, content)

    def add_assistant_message(self, content: str):
        self._hide_loading()
        self._notify_message(MessageRole.ASSISTANT, content)

    def add_thinking_message(self, content: str):
        self._hide_loading()
        self._notify_message(MessageRole.THINKING, content)

    def add_system_message(self, content: str):
        self._hide_loading()
        self._notify_message(MessageRole.SYSTEM, content)

    def add_tool_message(self, content: str):
        self._hide_loading()
        self._notify_message(MessageRole.TOOL, content)

    def add_error_message(self, content: str):
        self._hide_loading()
        self._notify_message(MessageRole.ERROR, content)

    def create_task(self, task_id: str, title: str) -> Task:
        task = self.state.add_task(task_id, title)
        self._notify_task_update(task)
        return task

    def update_task(self, task_id: str, status: TaskStatus):
        task = self.state.update_task_status(task_id, status)
        if task:
            self._notify_task_update(task)

    def set_input_enabled(self, enabled: bool):
        self.state.input_enabled = enabled
        self._notify_state_change()

    async def start_research(self, query: str):
        if self.state.is_researching:
            return

        self.state.is_researching = True
        self.state.current_query = query
        self.state.clear_tasks()
        if self._on_tasks_clear:
            self._on_tasks_clear()
        self.set_input_enabled(False)

        self.add_user_message(query)

        # Show loading
        self._show_loading("Planning research")

        try:
            def on_event(event_type: str, data: dict):
                self._handle_event(event_type, data)

            report = await run_research_with_tools(query, callback=on_event)

            self._hide_loading()

            # Complete all remaining tasks
            for task in self.state.tasks:
                if task.status != TaskStatus.COMPLETED:
                    self.update_task(task.id, TaskStatus.COMPLETED)

            self.add_assistant_message(report)

        except Exception as e:
            self._hide_loading()
            self.add_error_message(f"{str(e)}")

        finally:
            self._hide_loading()
            self.state.is_researching = False
            self.set_input_enabled(True)
            self._notify_state_change()

    def _handle_event(self, event_type: str, data: dict):
        """Handle events from the research agent."""
        debug_log(f"Event: {event_type}, Data: {data}")

        if event_type == "node_start":
            node = data.get("node", "")
            self._handle_node_start(node, data)

        elif event_type == "hierarchical_plan_created":
            self._handle_hierarchical_plan(data)

        elif event_type == "task_progress":
            self._handle_task_progress(data)

        elif event_type == "plan_created":
            # Only handle if hierarchical plan wasn't already created
            if not self.state.tasks:
                queries = data.get("queries", [])
                self._handle_plan_created(queries)

        elif event_type == "query_completed":
            completed = data.get("completed", [])
            self._handle_query_completed(completed)

        elif event_type == "tool_call":
            tool_name = data.get("tool", "")
            args = data.get("args", {})
            self._handle_tool_call(tool_name, args)

        elif event_type == "tool_result":
            tool_name = data.get("tool", "")
            result = data.get("result", "")
            self._handle_tool_result(tool_name, result)

        elif event_type == "thinking":
            msg = data.get("message", "")
            self.add_thinking_message(msg)

        elif event_type == "llm_start":
            msg = data.get("message", "LLM responding")
            self._show_loading(msg)

    def _handle_node_start(self, node: str, data: dict):
        # Show loading for LLM nodes
        loading_msgs = {
            "plan": "Creating plan",
            "research": "Analyzing",
            "synthesize": "Writing report",
        }
        if node in loading_msgs:
            self._show_loading(loading_msgs[node])

    def _handle_plan_created(self, queries: list):
        """Handle research plan creation - add queries as tasks."""
        debug_log(f"_handle_plan_created called with {len(queries)} queries")
        debug_log(f"Current tasks: {len(self.state.tasks)}")

        # Skip if already have tasks (avoid duplicates)
        if self.state.tasks:
            debug_log("Skipping - tasks already exist")
            return

        self._hide_loading()

        for i, query in enumerate(queries):
            task_id = f"query_{i}"
            # Truncate long queries for display
            title = query if len(query) <= 50 else query[:47] + "..."
            debug_log(f"Creating task: {task_id} - {title}")
            self.create_task(task_id, title)

        # Mark first query as in progress
        if queries:
            self.update_task("query_0", TaskStatus.IN_PROGRESS)

        self.add_system_message(f"Research plan: {len(queries)} queries")

    def _handle_query_completed(self, completed: list):
        """Mark completed queries in task list."""
        for i, query in enumerate(completed):
            task_id = f"query_{i}"
            existing = next((t for t in self.state.tasks if t.id == task_id), None)
            if existing and existing.status != TaskStatus.COMPLETED:
                self.update_task(task_id, TaskStatus.COMPLETED)

        # Mark next query as in progress
        next_idx = len(completed)
        next_task_id = f"query_{next_idx}"
        next_task = next((t for t in self.state.tasks if t.id == next_task_id), None)
        if next_task and next_task.status == TaskStatus.PENDING:
            self.update_task(next_task_id, TaskStatus.IN_PROGRESS)

    def _handle_tool_call(self, tool_name: str, args: dict):
        self._hide_loading()
        # Build call text
        if tool_name == "zai-web-search":
            query = args.get("query", "")
            call_text = f"Searching: {query}"
        elif tool_name == "read_webpage":
            url = args.get("url", "")
            short_url = url[:60] + "..." if len(url) > 60 else url
            call_text = f"Reading: {short_url}"
        else:
            call_text = f"Calling: {tool_name}"

        # Use new callback to create tool widget
        if self._on_tool_call:
            widget_id = self._on_tool_call(tool_name, call_text)
            self._current_tool_widget_id = widget_id
            debug_log(f"Tool call widget created: {widget_id}")
        else:
            self.add_tool_message(call_text)

    def _handle_tool_result(self, tool_name: str, result: str):
        debug_log(f"_handle_tool_result called: tool={tool_name}, result={result[:100] if result else 'None'}...")
        self._hide_loading()

        # Determine result text and error status
        is_error = False
        if result is None:
            result_text = "(no response)"
        elif not result or not result.strip():
            result_text = "(empty)"
        else:
            result_lower = result.lower()
            if "error" in result_lower or "failed" in result_lower or "exception" in result_lower:
                is_error = True
                result_text = result[:200] + "..." if len(result) > 200 else result
            else:
                result_text = result[:150] + "..." if len(result) > 150 else result
            result_text = result_text.replace("\n", " ")

        # Use new callback to update tool widget
        if self._on_tool_result and self._current_tool_widget_id:
            self._on_tool_result(self._current_tool_widget_id, result_text, is_error)
            debug_log(f"Tool result updated: {self._current_tool_widget_id} -> {result_text[:50]}")
            self._current_tool_widget_id = None
        else:
            # Fallback to old method
            if is_error:
                self.add_error_message(f"Tool error: {result_text}")
            else:
                self.add_tool_message(f"Result: {result_text}")

    def _handle_hierarchical_plan(self, data: dict):
        """Create hierarchical task list from plan."""
        debug_log(f"_handle_hierarchical_plan called with {data}")

        # Skip if already have tasks (avoid duplicates)
        if self.state.tasks:
            debug_log("Skipping - tasks already exist")
            return

        self._hide_loading()

        main_tasks = data.get("main_tasks", [])
        intent_count = data.get("intent_count", len(main_tasks))

        for main_task in main_tasks:
            main_id = f"main_{main_task['id']}"
            # Use ● marker instead of [] to avoid Rich markup conflicts
            topic = main_task['topic'][:35]
            main_title = f"● {topic}"
            debug_log(f"Creating main task: {main_id} - {main_title}")
            self.create_task(main_id, main_title)

            # Create sub-tasks with indentation
            for sub_task in main_task.get("sub_tasks", []):
                sub_id = f"sub_{sub_task['id']}"
                query = sub_task["query"]
                sub_title = f"  └ {query[:38]}" if len(query) <= 38 else f"  └ {query[:35]}..."
                debug_log(f"Creating sub task: {sub_id} - {sub_title}")
                self.create_task(sub_id, sub_title)

        total_tasks = sum(len(mt.get("sub_tasks", [])) for mt in main_tasks)
        self.add_system_message(f"Plan: {intent_count} topics, {total_tasks} queries")

    def _handle_task_progress(self, data: dict):
        """Update task progress for hierarchical tasks."""
        main_id = data.get("main_task_id")
        sub_id = data.get("sub_task_id")
        status = data.get("status", "in_progress")

        debug_log(f"Task progress: main={main_id}, sub={sub_id}, status={status}")

        if status == "in_progress":
            # Update main task to in_progress
            self.update_task(f"main_{main_id}", TaskStatus.IN_PROGRESS)
            # Update sub-task to in_progress
            self.update_task(f"sub_{sub_id}", TaskStatus.IN_PROGRESS)
        elif status == "completed":
            # Mark sub-task as completed
            self.update_task(f"sub_{sub_id}", TaskStatus.COMPLETED)

    def cancel_research(self):
        self._hide_loading()
        if self._research_task and not self._research_task.done():
            self._research_task.cancel()
            self.state.is_researching = False
            self.set_input_enabled(True)
            self.add_system_message("Research cancelled.")
