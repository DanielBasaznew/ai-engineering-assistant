"""
Main entry point for AI Engineering Assistant (Production Ready).
Run:
    python main.py
"""

import sys
from assistant import Assistant


def main():
    try:
        assistant = Assistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\n[Application interrupted by user. Exiting cleanly.]")
    except Exception as e:
        print(f"\n[Fatal Error]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()