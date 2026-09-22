"""
Main entry point for AI Engineering Assistant (Production Ready).
Run:
    python main.py
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich import box
from assistant import Assistant

console = Console()


def main():
    try:
        assistant = Assistant()
        assistant.run()
    except KeyboardInterrupt:
        console.print("\n[bold yellow][!] Session interrupted by user. Exiting cleanly.[/bold yellow]")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]Fatal Startup Error:[/bold red] {e}",
                title="System Error",
                border_style="red",
                box=box.ROUNDED,
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()