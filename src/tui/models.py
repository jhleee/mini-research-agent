"""Data models for the TUI."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class MessageRole(Enum):
    """Role of a chat message."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    THINKING = "thinking"
    TOOL = "tool"
    ERROR = "error"


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class ChatMessage:
    """A chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def get_css_class(self) -> str:
        """Get the CSS class for this message type."""
        return f"{self.role.value}-message"


@dataclass
class Task:
    """A task item."""
    id: str
    title: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def get_css_class(self) -> str:
        """Get the CSS class for this task status."""
        return f"task-{self.status.value.replace('_', '-')}"

    def get_status_icon(self) -> str:
        """Get the status icon."""
        icons = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.IN_PROGRESS: "[~]",
            TaskStatus.COMPLETED: "[x]",
        }
        return icons.get(self.status, "[ ]")


@dataclass
class AppState:
    """Application state."""
    messages: list[ChatMessage] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    is_researching: bool = False
    current_query: str = ""
    input_enabled: bool = True

    def add_message(self, role: MessageRole, content: str) -> ChatMessage:
        """Add a new message."""
        msg = ChatMessage(role=role, content=content)
        self.messages.append(msg)
        return msg

    def add_task(self, task_id: str, title: str) -> Task:
        """Add a new task."""
        task = Task(id=task_id, title=title)
        self.tasks.append(task)
        return task

    def update_task_status(self, task_id: str, status: TaskStatus) -> Optional[Task]:
        """Update a task's status."""
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                if status == TaskStatus.COMPLETED:
                    task.completed_at = datetime.now()
                return task
        return None

    def get_active_tasks(self) -> list[Task]:
        """Get tasks that are not completed."""
        return [t for t in self.tasks if t.status != TaskStatus.COMPLETED]

    def get_completed_tasks(self) -> list[Task]:
        """Get completed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def clear_tasks(self):
        """Clear all tasks."""
        self.tasks.clear()
