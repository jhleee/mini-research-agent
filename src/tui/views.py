"""View components for the Research Agent TUI."""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Input, Button
from textual.widget import Widget
from textual.message import Message

from .models import ChatMessage, Task, TaskStatus, MessageRole

# Debug logging to file
DEBUG_LOG = open("debug.log", "a", encoding="utf-8")

def debug_log(msg: str):
    DEBUG_LOG.write(f"[views] {msg}\n")
    DEBUG_LOG.flush()


class MessageWidget(Static):
    """A single chat message widget."""

    def __init__(self, message: ChatMessage, **kwargs):
        role_prefix = {
            MessageRole.USER: "[bold cyan]>[/bold cyan]",
            MessageRole.ASSISTANT: "[bold green]Assistant:[/bold green]",
            MessageRole.SYSTEM: "[dim]#[/dim]",
            MessageRole.THINKING: "[yellow]...[/yellow]",
            MessageRole.TOOL: "[magenta]Tool:[/magenta]",
            MessageRole.ERROR: "[bold red]Error:[/bold red]",
        }
        prefix = role_prefix.get(message.role, "")
        content = message.content

        if message.role == MessageRole.THINKING and len(content) > 100:
            content = content[:100] + "..."

        # Wrap error content in red
        if message.role == MessageRole.ERROR:
            content = f"[red]{content}[/red]"

        super().__init__(f"{prefix} {content}", **kwargs)
        self.add_class("chat-message")
        self.add_class(message.get_css_class())


class ToolExecutionWidget(Static):
    """Widget that shows tool execution with call and result together."""

    def __init__(self, tool_name: str, call_text: str, **kwargs):
        super().__init__("", **kwargs)
        self.tool_name = tool_name
        self.call_text = call_text
        self.result_text = None
        self.is_error = False
        self.add_class("chat-message")
        self.add_class("tool-execution")
        self._update_display()

    def _update_display(self):
        """Update the widget display based on current state."""
        lines = [f"[magenta]Tool:[/magenta] {self.call_text}"]

        if self.result_text is None:
            # Still waiting for result
            lines.append("[dim]  ⏳ waiting...[/dim]")
        elif self.is_error:
            lines.append(f"[red]  ✗ {self.result_text}[/red]")
        else:
            lines.append(f"[dim]  → {self.result_text}[/dim]")

        self.update("\n".join(lines))

    def set_result(self, result: str, is_error: bool = False):
        """Set the result and update display."""
        self.result_text = result
        self.is_error = is_error
        self._update_display()


class TaskWidget(Static):
    """A single task widget."""

    STATUS_STYLES = {
        TaskStatus.PENDING: ("[dim][   ][/dim]", "[dim]"),
        TaskStatus.IN_PROGRESS: ("[yellow][...][/yellow]", "[yellow]"),
        TaskStatus.COMPLETED: ("[green][ * ][/green]", "[green]"),
    }

    def __init__(self, task: Task, **kwargs):
        super().__init__(self._format_task(task), **kwargs)
        self._task_data = task
        self.add_class("task-item")

    def _format_task(self, task: Task) -> str:
        """Format task content with status icon and color."""
        icon, color = self.STATUS_STYLES.get(task.status, ("[dim][  ][/dim]", "[dim]"))
        return f"{icon} {color}{task.title}[/]"

    def update_task(self, task: Task):
        """Update the widget with new task data in place."""
        self._task_data = task
        self.update(self._format_task(task))


class LoadingWidget(Static):
    """Loading indicator with animated dots."""

    def __init__(self, text: str = "LLM responding", **kwargs):
        super().__init__("", **kwargs)
        self.text = text
        self._dot_count = 0
        self.add_class("loading-widget")

    def on_mount(self):
        self._update_display()
        self.set_interval(0.3, self._update_dots)

    def _update_display(self):
        dots = "." * (self._dot_count + 1)
        self.update(f"[bold yellow]⏳ {self.text}{dots}[/bold yellow]")

    def _update_dots(self):
        self._dot_count = (self._dot_count + 1) % 3
        self._update_display()


class ChatPanel(Widget):
    """Left panel containing the chat messages."""

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-scroll")

    def add_message(self, message: ChatMessage):
        try:
            # Remove loading indicator if present
            self.hide_loading()

            scroll = self.query_one("#chat-scroll", VerticalScroll)
            widget = MessageWidget(message, id=f"msg-{len(scroll.children)}")
            scroll.mount(widget)
            scroll.scroll_end(animate=False)
        except Exception as e:
            debug_log(f"ChatPanel.add_message ERROR: {e}")

    def add_tool_execution(self, tool_name: str, call_text: str) -> str:
        """Add a tool execution widget and return its ID."""
        try:
            self.hide_loading()
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            widget_id = f"tool-{len(scroll.children)}"
            widget = ToolExecutionWidget(tool_name, call_text, id=widget_id)
            scroll.mount(widget)
            scroll.scroll_end(animate=False)
            return widget_id
        except Exception as e:
            debug_log(f"ChatPanel.add_tool_execution ERROR: {e}")
            return ""

    def update_tool_result(self, widget_id: str, result: str, is_error: bool = False):
        """Update a tool execution widget with its result."""
        try:
            if not widget_id:
                return
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            widget = scroll.query_one(f"#{widget_id}", ToolExecutionWidget)
            widget.set_result(result, is_error)
        except Exception as e:
            debug_log(f"ChatPanel.update_tool_result ERROR: {e}")

    def show_loading(self, text: str = "LLM responding"):
        try:
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            # Check if loading already shown
            try:
                scroll.query_one("#loading-indicator")
                return
            except Exception:
                pass
            loading = LoadingWidget(text, id="loading-indicator")
            scroll.mount(loading)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def hide_loading(self):
        try:
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            loading = scroll.query_one("#loading-indicator")
            loading.remove()
        except Exception:
            pass

    def clear_messages(self):
        try:
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            scroll.remove_children()
        except Exception:
            pass


class TaskPanel(Widget):
    """Right panel containing task list."""

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="task-list")

    def _sanitize_id(self, task_id: str) -> str:
        """Convert task ID to valid Textual widget ID.

        Textual IDs must contain only letters, numbers, underscores, or hyphens.
        Replace dots with dashes.
        """
        return task_id.replace(".", "-")

    def add_task(self, task: Task):
        from .controller import debug_log
        try:
            sanitized_id = self._sanitize_id(task.id)
            task_list = self.query_one("#task-list", VerticalScroll)

            # Check if widget already exists
            try:
                existing = task_list.query_one(f"#task-{sanitized_id}", TaskWidget)
                # Widget exists, update it in place
                existing.update_task(task)
                debug_log(f"TaskPanel.add_task: updated existing {task.id}")
                return
            except Exception:
                pass

            # Widget doesn't exist, create new one
            widget = TaskWidget(task, id=f"task-{sanitized_id}")
            task_list.mount(widget)
            debug_log(f"TaskPanel.add_task: mounted {task.id}")
        except Exception as e:
            debug_log(f"TaskPanel.add_task error: {e}")

    def update_task(self, task: Task):
        """Update task in place without changing position."""
        from .controller import debug_log
        try:
            sanitized_id = self._sanitize_id(task.id)
            widget = self.query_one(f"#task-{sanitized_id}", TaskWidget)
            widget.update_task(task)
            debug_log(f"TaskPanel.update_task: updated {task.id}")
        except Exception:
            # Widget doesn't exist yet, add it
            self.add_task(task)

    def clear_tasks(self):
        try:
            task_list = self.query_one("#task-list", VerticalScroll)
            task_list.remove_children()
        except Exception:
            pass


class InputBar(Widget):
    """Input area with text field and send button."""

    class Submitted(Message):
        def __init__(self, value: str):
            super().__init__()
            self.value = value

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter your research question...", id="user-input")
        yield Button("Send", id="send-button", variant="primary")

    def on_mount(self):
        self.call_after_refresh(self._focus_input)

    def _focus_input(self):
        try:
            self.query_one("#user-input", Input).focus()
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted):
        if event.value.strip():
            self.post_message(self.Submitted(event.value.strip()))
            event.input.clear()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "send-button":
            try:
                input_widget = self.query_one("#user-input", Input)
                if input_widget.value.strip():
                    self.post_message(self.Submitted(input_widget.value.strip()))
                    input_widget.clear()
            except Exception:
                pass

    def set_enabled(self, enabled: bool):
        try:
            input_widget = self.query_one("#user-input", Input)
            send_button = self.query_one("#send-button", Button)
            input_widget.disabled = not enabled
            send_button.disabled = not enabled
            if enabled:
                input_widget.focus()
        except Exception:
            pass


class Header(Static):
    def __init__(self, title: str = "Deep Research Agent", **kwargs):
        super().__init__(title, **kwargs)


class Footer(Static):
    def __init__(self, text: str = "Ctrl+Q: Quit", **kwargs):
        super().__init__(text, **kwargs)
