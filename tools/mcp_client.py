"""
MCP (Model Context Protocol) Client Integration.
Connects the Assistant to the Week 8 Enhanced MCP Server via stdio JSON-RPC transport.
"""

import asyncio
import os
import sys
from typing import Any, Dict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from logger import log


def get_mcp_server_params() -> StdioServerParameters:
    """
    Resolves the execution parameters for the Week 8 MCP server.
    """
    week8_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mcp-coding-assistant-week8"))
    py312 = os.path.join(week8_dir, "venv", "Scripts", "python.exe")
    server_path = os.path.join(week8_dir, "enhanced_server.py")

    cmd = py312 if os.path.exists(py312) else sys.executable
    return StdioServerParameters(command=cmd, args=[server_path])


async def _async_call_mcp_tool(name: str, arguments: Dict[str, Any]) -> str:
    params = get_mcp_server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            if result.content:
                return result.content[0].text
            return "MCP tool completed with no output."


def call_mcp_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    Executes a tool on the Week 8 MCP server synchronously over stdio.
    """
    try:
        return asyncio.run(_async_call_mcp_tool(name, arguments))
    except Exception as e:
        log.error(f"MCP tool call '{name}' failed: {e}")
        return f"Error executing MCP tool '{name}': {str(e)}"
