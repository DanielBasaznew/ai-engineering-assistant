"""
Python Code Execution Tool (Week 4, Day 3)
Safely runs agent-generated Python code in a subprocess with timeouts and caps.
"""

import os
import subprocess
import sys
import tempfile


def execute_python(code: str) -> str:
    """
    Executes a string of Python code in an isolated subprocess.
    Returns the combined stdout and stderr output capped at 2000 characters.
    """
    # 1. Write the code string to a temporary file in the system temp directory
    temp_dir = tempfile.gettempdir()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=temp_dir, delete=False, encoding="utf-8"
    ) as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name

    try:
        # 2. Run the code in a separate subprocess with a 15-second timeout
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=temp_dir,  # Set working directory to temp dir, keeping main project clean
        )

        # 3. Combine standard output and error messages
        output = result.stdout
        if result.stderr:
            output += f"\n[Errors/Warnings]:\n{result.stderr}"

        if not output.strip():
            output = (
                "Code executed successfully with no printed output. "
                "Note: Ensure your script uses print() to display results or save outputs."
            )

        # Cap output to 2000 characters to protect context window
        return output[:2000]

    except subprocess.TimeoutExpired:
        return (
            "Error: Code execution timed out after 15 seconds. "
            "Ensure your code does not contain infinite loops."
        )
    except Exception as e:
        return f"Error executing Python code: {str(e)}"
    finally:
        # 4. Clean up: always remove the temporary file regardless of success or failure
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print(execute_python("print('Hello from isolated python executor!')"))
