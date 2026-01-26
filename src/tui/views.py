"""View components for the Research Agent TUI."""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Input, Button
from textual.widget import Widget
from textual.message import Message

from .models import ChatMessage, Task, TaskStatus, MessageRole


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


class TaskWidget(Static):
    """A single task widget."""

    def __init__(self, task: Task, **kwargs):
        # Color-coded checkbox style based on status
        status_styles = {
            TaskStatus.PENDING: ("[dim][  ][/dim]", "[dim]"),
            TaskStatus.IN_PROGRESS: ("[yellow][..][/yellow]", "[yellow]"),
            TaskStatus.COMPLETED: ("[green][*][/green]", "[green]"),
        }
        icon, color = status_styles.get(task.status, ("[dim][  ][/dim]", "[dim]"))
        content = f"{icon} {color}{task.title}[/]"
        super().__init__(content, **kwargs)
        self._task_data = task
        self.add_class("task-item")


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
        except Exception:
            pass

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

    def add_task(self, task: Task):
        from .controller import debug_log
        try:
            task_list = self.query_one("#task-list", VerticalScroll)
            widget = TaskWidget(task, id=f"task-{task.id}")
            task_list.mount(widget)
            debug_log(f"TaskPanel.add_task: mounted {task.id}")
        except Exception as e:
            debug_log(f"TaskPanel.add_task error: {e}")

    def update_task(self, task: Task):
        try:
            widget = self.query_one(f"#task-{task.id}", TaskWidget)
            widget.remove()
        except Exception:
            pass
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
