#!/usr/bin/env python3
"""Run the Deep Research Agent."""
import sys


def main():
    """Main entry point."""
    # Check for CLI mode flag
    if "--cli" in sys.argv or "-c" in sys.argv:
        # Remove the flag and run CLI mode
        sys.argv = [arg for arg in sys.argv if arg not in ("--cli", "-c")]
        from src.cli import main as cli_main
        cli_main()
    else:
        # Run TUI mode (default)
        from src.tui.app import run_app
        run_app()


if __name__ == "__main__":
    main()
