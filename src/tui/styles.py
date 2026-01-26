"""CSS Styles for the Research Agent TUI."""

APP_CSS = """
Screen {
    layout: vertical;
    background: $surface;
}

/* Loading screen */
#loading-container {
    width: 100%;
    height: 100%;
    align: center middle;
}

#loading-text {
    width: auto;
    height: auto;
    padding: 2 4;
    border: round $primary;
    text-align: center;
}

/* Main layout containers */
#main-container {
    width: 100%;
    height: 1fr;
    layout: horizontal;
    padding: 1;
}

#left-panel {
    width: 2fr;
    height: 100%;
    layout: vertical;
    padding: 0 1 0 0;
}

#right-panel {
    width: 1fr;
    height: 100%;
    min-width: 35;
    layout: vertical;
    border: round $surface-lighten-2;
    padding: 1;
}

/* Chat Panel - Left side */
#chat-panel {
    width: 100%;
    height: 1fr;
    border: round $surface-lighten-2;
    padding: 1;
}

#chat-scroll {
    width: 100%;
    height: 1fr;
    scrollbar-size: 1 1;
}

.chat-message {
    width: 100%;
    padding: 0;
    margin: 0 0 0 0;
}

.user-message {
    color: $text;
}

.assistant-message {
    color: $success;
}

.thinking-message {
    color: $warning;
}

.system-message {
    color: $text-muted;
}

.tool-message {
    color: $accent;
}

/* Loading widget */
.loading-widget {
    width: 100%;
    height: auto;
    padding: 0;
    margin: 0 0 1 0;
    color: $warning;
}

/* Input Area */
#input-area {
    width: 100%;
    height: auto;
    min-height: 3;
    max-height: 5;
    layout: horizontal;
    padding: 1 0 0 0;
}

#user-input {
    width: 1fr;
    height: 3;
    border: round $primary;
}

#user-input:focus {
    border: round $accent;
}

#user-input.-disabled {
    border: round $surface-lighten-1;
}

#send-button {
    width: 10;
    height: 3;
    margin: 0 0 0 1;
    background: $primary;
    color: $text;
    border: none;
}

#send-button:hover {
    background: $primary-lighten-1;
}

#send-button.-disabled {
    background: $surface-lighten-1;
    color: $text-muted;
}

/* Task Panel - Right side */
#task-header {
    width: 100%;
    height: 1;
    text-style: bold;
    margin: 0 0 1 0;
}

#task-list {
    width: 100%;
    height: 1fr;
    scrollbar-size: 1 1;
}

.task-item {
    width: 100%;
    height: 1;
    padding: 0;
    margin: 0;
}

.task-pending {
    color: $text-muted;
}

.task-in-progress {
    color: $warning;
}

.task-completed {
    color: $success;
}

/* Header */
#header {
    width: 100%;
    height: 3;
    background: $primary;
    color: $text;
    content-align: center middle;
    text-style: bold;
    dock: top;
}

/* Footer */
#footer {
    width: 100%;
    height: 1;
    background: $surface-darken-1;
    color: $text-muted;
    content-align: center middle;
    dock: bottom;
}
"""
