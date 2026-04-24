"""
MCP Integration for GAMI.

Enables GAMI to work with MCP (Model Context Protocol) tools and Claude Code.
Supports:
- Database operations via MCP tools
- Claude Code as an LLM provider
- Tool execution and result handling
- Session management
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum

import httpx

from .providers import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    BaseLLMProvider,
    ProviderConfig,
    ProviderRegistry,
)

logger = logging.getLogger("gami.llm.mcp")


# =============================================================================
# MCP TOOL DEFINITIONS
# =============================================================================

@dataclass
class MCPTool:
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None


@dataclass
class MCPToolResult:
    """Result from an MCP tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


@dataclass
class MCPMessage:
    """Message in MCP format."""
    role: str  # "user", "assistant", "tool_result"
    content: Union[str, List[Dict]]


# =============================================================================
# DATABASE MCP TOOLS
# =============================================================================

class DatabaseMCPTools:
    """
    MCP tools for database operations.

    These tools allow Claude Code or other MCP clients to:
    - Query the database
    - Create/update tables
    - Run migrations
    - Manage data
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self._engine = None

    async def _get_engine(self):
        """Get or create async database engine."""
        if self._engine is None:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._engine = create_async_engine(self.db_url)
        return self._engine

    def get_tools(self) -> List[MCPTool]:
        """Return list of available database tools."""
        return [
            MCPTool(
                name="db_query",
                description="Execute a read-only SQL query and return results",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT query to execute",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum rows to return (default 100)",
                            "default": 100,
                        },
                    },
                    "required": ["query"],
                },
                handler=self.execute_query,
            ),
            MCPTool(
                name="db_execute",
                description="Execute a SQL statement (INSERT, UPDATE, DELETE, CREATE, ALTER)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "SQL statement to execute",
                        },
                        "params": {
                            "type": "object",
                            "description": "Parameters for parameterized query",
                        },
                    },
                    "required": ["statement"],
                },
                handler=self.execute_statement,
            ),
            MCPTool(
                name="db_schema",
                description="Get schema information for a table",
                input_schema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table",
                        },
                    },
                    "required": ["table_name"],
                },
                handler=self.get_schema,
            ),
            MCPTool(
                name="db_tables",
                description="List all tables in the database",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
                handler=self.list_tables,
            ),
            MCPTool(
                name="db_migrate",
                description="Run a SQL migration file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL migration to execute",
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the migration",
                        },
                    },
                    "required": ["sql"],
                },
                handler=self.run_migration,
            ),
        ]

    async def execute_query(self, query: str, limit: int = 100) -> MCPToolResult:
        """Execute a read-only SQL query."""
        try:
            # Ensure it's a SELECT query
            query_upper = query.strip().upper()
            if not query_upper.startswith("SELECT") and not query_upper.startswith("WITH"):
                return MCPToolResult(
                    tool_name="db_query",
                    success=False,
                    result=None,
                    error="Only SELECT queries allowed. Use db_execute for modifications.",
                )

            engine = await self._get_engine()
            from sqlalchemy import text

            async with engine.connect() as conn:
                result = await conn.execute(text(f"{query} LIMIT {limit}"))
                rows = result.fetchall()
                columns = list(result.keys())

                # Convert to list of dicts
                data = [dict(zip(columns, row)) for row in rows]

                return MCPToolResult(
                    tool_name="db_query",
                    success=True,
                    result={"columns": columns, "rows": data, "count": len(data)},
                )

        except Exception as e:
            return MCPToolResult(
                tool_name="db_query",
                success=False,
                result=None,
                error=str(e),
            )

    async def execute_statement(
        self, statement: str, params: Optional[Dict] = None
    ) -> MCPToolResult:
        """Execute a SQL statement (INSERT, UPDATE, DELETE, etc.)."""
        try:
            engine = await self._get_engine()
            from sqlalchemy import text

            async with engine.begin() as conn:
                result = await conn.execute(text(statement), params or {})
                rowcount = result.rowcount

                return MCPToolResult(
                    tool_name="db_execute",
                    success=True,
                    result={"rowcount": rowcount, "statement": statement[:100]},
                )

        except Exception as e:
            return MCPToolResult(
                tool_name="db_execute",
                success=False,
                result=None,
                error=str(e),
            )

    async def get_schema(self, table_name: str) -> MCPToolResult:
        """Get schema information for a table."""
        try:
            engine = await self._get_engine()
            from sqlalchemy import text

            async with engine.connect() as conn:
                # Get columns
                result = await conn.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """), {"table_name": table_name})
                columns = [dict(zip(result.keys(), row)) for row in result.fetchall()]

                # Get indexes
                result = await conn.execute(text("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = :table_name
                """), {"table_name": table_name})
                indexes = [dict(zip(result.keys(), row)) for row in result.fetchall()]

                return MCPToolResult(
                    tool_name="db_schema",
                    success=True,
                    result={"table": table_name, "columns": columns, "indexes": indexes},
                )

        except Exception as e:
            return MCPToolResult(
                tool_name="db_schema",
                success=False,
                result=None,
                error=str(e),
            )

    async def list_tables(self) -> MCPToolResult:
        """List all tables in the database."""
        try:
            engine = await self._get_engine()
            from sqlalchemy import text

            async with engine.connect() as conn:
                result = await conn.execute(text("""
                    SELECT table_name,
                           pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """))
                tables = [dict(zip(result.keys(), row)) for row in result.fetchall()]

                return MCPToolResult(
                    tool_name="db_tables",
                    success=True,
                    result={"tables": tables, "count": len(tables)},
                )

        except Exception as e:
            return MCPToolResult(
                tool_name="db_tables",
                success=False,
                result=None,
                error=str(e),
            )

    async def run_migration(
        self, sql: str, description: Optional[str] = None
    ) -> MCPToolResult:
        """Run a SQL migration."""
        try:
            engine = await self._get_engine()
            from sqlalchemy import text

            async with engine.begin() as conn:
                # Execute the migration
                await conn.execute(text(sql))

                # Log the migration
                try:
                    await conn.execute(text("""
                        INSERT INTO schema_migrations (description, sql_hash, applied_at)
                        VALUES (:desc, :hash, NOW())
                    """), {
                        "desc": description or "Manual migration",
                        "hash": hash(sql) & 0xFFFFFFFF,
                    })
                except Exception:
                    # Migration log table may not exist
                    pass

                return MCPToolResult(
                    tool_name="db_migrate",
                    success=True,
                    result={"description": description, "applied": True},
                )

        except Exception as e:
            return MCPToolResult(
                tool_name="db_migrate",
                success=False,
                result=None,
                error=str(e),
            )


# =============================================================================
# MCP SERVER CLIENT
# =============================================================================

class MCPClient:
    """
    Client for connecting to MCP servers.

    Can connect to:
    - Local MCP servers (stdio)
    - Remote MCP servers (HTTP/SSE)
    - GAMI's own MCP server
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        server_command: Optional[List[str]] = None,
    ):
        """
        Initialize MCP client.

        Args:
            server_url: URL for HTTP/SSE MCP server
            server_command: Command to start stdio MCP server
        """
        self.server_url = server_url
        self.server_command = server_command
        self._process: Optional[subprocess.Popen] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._tools: Dict[str, MCPTool] = {}

    async def connect(self):
        """Connect to the MCP server."""
        if self.server_url:
            self._http_client = httpx.AsyncClient(timeout=60.0)
            # Fetch available tools
            await self._fetch_tools()
        elif self.server_command:
            # Start stdio server
            self._process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    async def _fetch_tools(self):
        """Fetch available tools from HTTP MCP server."""
        if not self._http_client or not self.server_url:
            return

        try:
            response = await self._http_client.get(f"{self.server_url}/tools")
            response.raise_for_status()
            tools_data = response.json()

            for tool in tools_data.get("tools", []):
                self._tools[tool["name"]] = MCPTool(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                )
        except Exception as e:
            logger.warning(f"Failed to fetch MCP tools: {e}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call an MCP tool."""
        if self._http_client and self.server_url:
            return await self._call_tool_http(tool_name, arguments)
        elif self._process:
            return await self._call_tool_stdio(tool_name, arguments)
        else:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error="No MCP server connected",
            )

    async def _call_tool_http(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> MCPToolResult:
        """Call tool via HTTP."""
        try:
            response = await self._http_client.post(
                f"{self.server_url}/tools/{tool_name}",
                json=arguments,
            )
            response.raise_for_status()
            data = response.json()

            return MCPToolResult(
                tool_name=tool_name,
                success=data.get("success", True),
                result=data.get("result"),
                error=data.get("error"),
            )
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )

    async def _call_tool_stdio(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> MCPToolResult:
        """Call tool via stdio."""
        if not self._process:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error="No stdio process",
            )

        try:
            # Send request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()

            # Read response
            response_line = self._process.stdout.readline()
            response = json.loads(response_line)

            if "error" in response:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=response["error"].get("message", str(response["error"])),
                )

            result = response.get("result", {})
            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=result.get("content", []),
            )

        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )

    def get_available_tools(self) -> List[MCPTool]:
        """Get list of available tools."""
        return list(self._tools.values())

    async def close(self):
        """Close the connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        if self._process:
            self._process.terminate()
            self._process = None


# =============================================================================
# CLAUDE CODE PROVIDER
# =============================================================================

class ClaudeCodeProvider(BaseLLMProvider):
    """
    Use Claude Code CLI as an LLM provider.

    This allows GAMI to leverage Claude Code for:
    - Complex multi-step tasks
    - Code generation and editing
    - Database operations via MCP tools
    """

    provider_type = LLMProvider.ANTHROPIC  # Maps to Anthropic

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._claude_path = self._find_claude_code()

    def _find_claude_code(self) -> Optional[str]:
        """Find Claude Code CLI."""
        # Check common locations
        candidates = [
            "claude",  # In PATH
            "/usr/local/bin/claude",
            os.path.expanduser("~/.local/bin/claude"),
            os.path.expanduser("~/bin/claude"),
        ]

        for path in candidates:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return path
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run completion via Claude Code CLI."""
        import time
        start = time.time()

        if not self._claude_path:
            raise RuntimeError(
                "Claude Code CLI not found. Install from: https://claude.ai/claude-code"
            )

        # Build the prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"

        # Run Claude Code in non-interactive mode
        try:
            process = await asyncio.create_subprocess_exec(
                self._claude_path,
                "--print",  # Non-interactive, print result
                "--model", request.model or "claude-sonnet-4-20250514",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=full_prompt.encode()),
                timeout=self.config.timeout_seconds,
            )

            content = stdout.decode().strip()
            latency_ms = (time.time() - start) * 1000

            if process.returncode != 0:
                error_msg = stderr.decode().strip()
                raise RuntimeError(f"Claude Code failed: {error_msg}")

            return LLMResponse(
                content=content,
                model=request.model or "claude-sonnet-4-20250514",
                provider=self.provider_type,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Claude Code timed out after {self.config.timeout_seconds}s"
            )


# =============================================================================
# MCP-ENABLED COMPLETION
# =============================================================================

class MCPEnabledLLM:
    """
    LLM completion with MCP tool support.

    Wraps any LLM provider and adds ability to:
    - Use MCP tools during completion
    - Handle tool calls automatically
    - Chain tool results back to LLM
    """

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        mcp_tools: Optional[List[MCPTool]] = None,
        mcp_client: Optional[MCPClient] = None,
    ):
        self.llm_provider = llm_provider or ProviderRegistry.get_llm_provider()
        self.mcp_tools = {t.name: t for t in (mcp_tools or [])}
        self.mcp_client = mcp_client

        # Add database tools by default
        db_tools = DatabaseMCPTools()
        for tool in db_tools.get_tools():
            self.mcp_tools[tool.name] = tool

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for inclusion in prompt."""
        if not self.mcp_tools:
            return ""

        tools_desc = ["Available tools:"]
        for name, tool in self.mcp_tools.items():
            tools_desc.append(f"- {name}: {tool.description}")
            if tool.input_schema.get("properties"):
                props = tool.input_schema["properties"]
                params = ", ".join(
                    f"{k}: {v.get('type', 'any')}"
                    for k, v in props.items()
                )
                tools_desc.append(f"  Parameters: {params}")

        tools_desc.append(
            "\nTo use a tool, respond with JSON: "
            '{"tool": "tool_name", "arguments": {...}}'
        )

        return "\n".join(tools_desc)

    async def _execute_tool(self, tool_name: str, arguments: Dict) -> MCPToolResult:
        """Execute a tool by name."""
        if tool_name in self.mcp_tools:
            tool = self.mcp_tools[tool_name]
            if tool.handler:
                return await tool.handler(**arguments)

        if self.mcp_client:
            return await self.mcp_client.call_tool(tool_name, arguments)

        return MCPToolResult(
            tool_name=tool_name,
            success=False,
            result=None,
            error=f"Unknown tool: {tool_name}",
        )

    def _parse_tool_call(self, content: str) -> Optional[Dict]:
        """Parse tool call from LLM response."""
        # Look for JSON with tool key
        try:
            # Try direct parse
            data = json.loads(content.strip())
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        import re
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    async def complete_with_tools(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        **kwargs,
    ) -> LLMResponse:
        """
        Complete with automatic tool execution.

        Args:
            prompt: User prompt
            system_prompt: System prompt (tools will be appended)
            max_iterations: Max tool call iterations
            **kwargs: Additional LLM parameters

        Returns:
            Final LLM response after tool execution
        """
        # Build system prompt with tools
        tools_section = self._format_tools_for_prompt()
        full_system = system_prompt or ""
        if tools_section:
            full_system = f"{full_system}\n\n{tools_section}".strip()

        messages = [{"role": "user", "content": prompt}]
        final_response = None

        for i in range(max_iterations):
            # Build prompt from message history
            history_prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )

            request = LLMRequest(
                prompt=history_prompt,
                system_prompt=full_system,
                **kwargs,
            )

            response = await self.llm_provider.complete(request)
            final_response = response

            # Check for tool call
            tool_call = self._parse_tool_call(response.content)
            if not tool_call:
                # No tool call, we're done
                break

            # Execute tool
            tool_name = tool_call.get("tool")
            arguments = tool_call.get("arguments", {})

            logger.info(f"Executing tool: {tool_name} with {arguments}")
            result = await self._execute_tool(tool_name, arguments)

            # Add to message history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "tool_result",
                "content": json.dumps({
                    "tool": tool_name,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                }),
            })

        return final_response

    async def close(self):
        """Cleanup resources."""
        if self.mcp_client:
            await self.mcp_client.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def complete_with_mcp(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    tools: Optional[List[MCPTool]] = None,
    **kwargs,
) -> LLMResponse:
    """
    Complete with MCP tool support.

    Args:
        prompt: User prompt
        system_prompt: System prompt
        provider: LLM provider to use
        tools: Additional MCP tools to enable
        **kwargs: Additional parameters

    Returns:
        LLM response (may include tool execution results)
    """
    llm = ProviderRegistry.get_llm_provider(provider)
    mcp_llm = MCPEnabledLLM(llm_provider=llm, mcp_tools=tools)

    try:
        return await mcp_llm.complete_with_tools(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )
    finally:
        await mcp_llm.close()


def get_gami_mcp_tools() -> List[MCPTool]:
    """Get all GAMI MCP tools for database operations."""
    db_tools = DatabaseMCPTools()
    return db_tools.get_tools()
