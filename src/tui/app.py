"""Main Textual Application for Deep Research Agent."""
import asyncio
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, LoadingIndicator
from textual.binding import Binding
from textual.screen import Screen

from .styles import APP_CSS
from .views import ChatPanel, TaskPanel, InputBar, Header, Footer
from .controller import ResearchController
from .models import ChatMessage, Task, MessageRole
from ..config import OPENAI_MODEL

# Get model name, fallback to default if empty
MODEL_DISPLAY_NAME = OPENAI_MODEL if OPENAI_MODEL else "glm-4.7"


class LoadingScreen(Screen):
    """Loading screen shown during initialization."""

    def compose(self) -> ComposeResult:
        with Container(id="loading-container"):
            yield Static("Loading Deep Research Agent...", id="loading-text")
            yield LoadingIndicator()


class MainScreen(Screen):
    """Main application screen."""

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, controller: ResearchController):
        super().__init__()
        self.controller = controller

    def compose(self) -> ComposeResult:
        yield Header(f"Mini Research", id="header")

        with Container(id="main-container"):
            with Vertical(id="left-panel"):
                yield ChatPanel(id="chat-panel")
                yield InputBar(id="input-area")

            with Vertical(id="right-panel"):
                yield Static("[bold]Todo[/bold]", id="task-header")
                yield TaskPanel(id="task-panel")

        yield Footer("Ctrl+Q: Quit | Escape: Cancel", id="footer")

    def on_mount(self):
        """Called when screen is mounted."""
        self.controller.set_callbacks(
            on_message=self._on_new_message,
            on_task_update=self._on_task_update,
            on_state_change=self._on_state_change,
            on_loading=self._on_loading,
            on_tasks_clear=self._on_tasks_clear,
        )
        self.controller.add_system_message(
            "Welcome! Enter a research question to get started."
        )

    def _on_new_message(self, message: ChatMessage):
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        chat_panel.add_message(message)

    def _on_loading(self, show: bool, text: str):
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        if show:
            chat_panel.show_loading(text)
        else:
            chat_panel.hide_loading()

    def _on_task_update(self, task: Task):
        task_panel = self.query_one("#task-panel", TaskPanel)
        task_panel.update_task(task)

    def _on_state_change(self, state):
        input_bar = self.query_one("#input-area", InputBar)
        input_bar.set_enabled(state.input_enabled)

    def _on_tasks_clear(self):
        task_panel = self.query_one("#task-panel", TaskPanel)
        task_panel.clear_tasks()

    async def on_input_bar_submitted(self, event: InputBar.Submitted):
        if event.value:
            asyncio.create_task(self.controller.start_research(event.value))

    def action_cancel(self):
        self.controller.cancel_research()

    def action_quit(self):
        self.app.exit()


class ResearchApp(App):
    """Deep Research Agent TUI Application."""

    CSS = APP_CSS
    TITLE = "Deep Research Agent"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.controller = ResearchController()

    def on_mount(self):
        """Show loading screen, then switch to main."""
        self.push_screen(LoadingScreen())
        self.set_timer(0.5, self._show_main)

    def _show_main(self):
        """Switch to main screen."""
        self.pop_screen()
        self.push_screen(MainScreen(self.controller))

    def action_quit(self):
        self.exit()


def run_app():
    """Run the TUI application."""
    app = ResearchApp()
    app.run()


if __name__ == "__main__":
    run_app()
