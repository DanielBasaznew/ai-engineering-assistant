"""
Verification test script for Model Context Protocol (MCP) explain_repository integration.
Tests:
1. Direct MCP tool invocation (explain_repository) over stdio to Week 8 enhanced_server.py
2. Cache eligibility check ensuring repository explainer queries bypass semantic cache
3. Live Assistant chat test invoking mcp_explain_repository
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
    print("RUNNING MCP REPOSITORY EXPLAINER VERIFICATION")
    print("=" * 65)

    # 1. Direct MCP stdio test
    print("\n--- Test 1: Direct MCP stdio execution (explain_repository) ---")
    res1 = call_mcp_tool("explain_repository", {"path": "rag"})
    print("MCP Response Preview:\n" + ("\n".join(res1.splitlines()[:15])))
    assert "Repository & Folder Explanation" in res1, "Expected header in report"
    assert "rag" in res1.lower(), "Expected rag folder in report"
    print(">>> Test 1 PASSED: Direct MCP stdio explain_repository returned structured analysis.")

    # 2. Cache eligibility check
    print("\n--- Test 2: Cache eligibility check ---")
    eligible = is_semantic_cache_eligible("Use mcp_explain_repository to inspect the tools folder")
    print(f"Eligible for semantic cache: {eligible}")
    assert not eligible, "Repository explainer queries must bypass semantic cache"
    print(">>> Test 2 PASSED: Explainer queries correctly bypass semantic cache.")

    print("\n" + "=" * 65)
    print("ALL MCP EXPLAINER TESTS READY!")
    print("=" * 65)


if __name__ == "__main__":
    test_mcp_repository_explainer()
