"""
Verification test script for Model Context Protocol (MCP) explain_repository integration.
Tests:
1. Direct MCP tool invocation (explain_repository) on internal folder (e.g. 'rag')
2. Direct MCP tool invocation on sibling week folder (e.g. 'week 9' -> multi-agent-week9)
3. Direct MCP tool invocation on a single file (e.g. 'main.py')
4. Cache eligibility check ensuring repository explainer queries bypass semantic cache
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Force UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tools.mcp_client import call_mcp_tool
from cache import is_semantic_cache_eligible


def test_mcp_repository_explainer():
    print("=" * 65)
    print("RUNNING MCP REPOSITORY & FILE EXPLAINER VERIFICATION")
    print("=" * 65)

    # 1. Folder in current repo
    print("\n--- Test 1: Direct MCP execution on internal folder ('rag') ---")
    res1 = call_mcp_tool("explain_repository", {"path": "rag"})
    print("MCP Response Preview:\n" + ("\n".join(res1.splitlines()[:10])))
    assert "Repository & Folder Explanation" in res1, "Expected folder header"
    assert "rag" in res1.lower(), "Expected rag folder name in report"
    print(">>> Test 1 PASSED: Internal folder explained.")

    # 2. Sibling folder resolution ('week 9')
    print("\n--- Test 2: Sibling folder resolution ('week 9' -> 'multi-agent-week9') ---")
    res2 = call_mcp_tool("explain_repository", {"path": "week 9"})
    print("MCP Response Preview:\n" + ("\n".join(res2.splitlines()[:10])))
    assert "Repository & Folder Explanation" in res2, "Expected folder header"
    assert "multi-agent-week9" in res2.lower(), "Expected multi-agent-week9 in report"
    print(">>> Test 2 PASSED: External week 9 folder resolved and explained.")

    # 3. Single file inspection ('main.py')
    print("\n--- Test 3: Single file inspection ('main.py') ---")
    res3 = call_mcp_tool("explain_repository", {"path": "main.py"})
    print("MCP Response Preview:\n" + ("\n".join(res3.splitlines()[:10])))
    assert "File Explanation" in res3, "Expected file header"
    print(">>> Test 3 PASSED: Single file inspected.")

    # 4. Cache eligibility check
    print("\n--- Test 4: Cache eligibility check ---")
    eligible = is_semantic_cache_eligible("analysis week 9 folder by using mcp")
    print(f"Eligible for semantic cache: {eligible}")
    assert not eligible, "Repository explainer queries must bypass semantic cache"
    print(">>> Test 4 PASSED: Explainer queries correctly bypass semantic cache.")

    print("\n" + "=" * 65)
    print("ALL MCP EXPLAINER TESTS READY!")
    print("=" * 65)


if __name__ == "__main__":
    test_mcp_repository_explainer()
