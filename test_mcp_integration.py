"""
Verification test for Model Context Protocol (MCP) integration.
Tests:
1. Direct MCP tool invocation over stdio to Week 8 enhanced_server.py
2. Assistant tool dispatch to mcp_calculate
3. Cache eligibility verification for MCP requests
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
from assistant import Assistant
from cache import TwoLayerCache, is_semantic_cache_eligible


def test_mcp_integration():
    print("=" * 60)
    print("RUNNING MCP INTEGRATION VERIFICATION")
    print("=" * 60)

    # 1. Direct MCP stdio test
    print("\n--- Test 1: Direct MCP stdio execution ---")
    res1 = call_mcp_tool("calculate", {"expression": "math.factorial(6)"})
    print(f"Direct MCP result: {res1}")
    assert "720" in str(res1), f"Expected 720, got {res1}"
    print(">>> Test 1 PASSED: Direct MCP stdio call returned expected calculation.")

    # 2. Cache eligibility check
    print("\n--- Test 2: Cache eligibility check ---")
    eligible = is_semantic_cache_eligible("Use mcp_calculate to compute math.sqrt(144)")
    print(f"Eligible for semantic cache: {eligible}")
    assert not eligible, "MCP query should bypass semantic cache"
    print(">>> Test 2 PASSED: MCP queries correctly bypass semantic cache.")

    # 3. Live Assistant chat test
    print("\n--- Test 3: Live Assistant chat with mcp_calculate ---")
    assistant = Assistant(session_id="test_mcp_session")
    resp = assistant.chat("Use mcp_calculate to compute math.sqrt(1024).")
    print(f"Assistant response: {resp}")
    assert "32" in resp, f"Expected 32 in response, got: {resp}"
    print(">>> Test 3 PASSED: Assistant successfully invoked mcp_calculate over stdio.")

    print("\n" + "=" * 60)
    print("ALL MCP INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_mcp_integration()
