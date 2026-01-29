import asyncio
import os
import shutil
import logging
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Optional, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import sys


class MCPClient:
    """
    A generic client for interacting with MCP (Model Control Protocol) servers.
    Supports both Stdio and SSE transport protocols.
    """

    def __init__(self, name: str, timeout: int = 60):
        """
        Initialize the MCP client.

        Args:
            name: Client identifier for logging.
            timeout: Session read timeout in seconds.
        """
        self._name = name
        self._timeout = timeout
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self._cleanup_lock = asyncio.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)

    async def connect_to_stdio_server(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """
        Connect to an MCP server using Stdio transport.

        Args:
            command: Executable command (e.g., 'npx', 'python').
            args: Command arguments.
            env: Additional environment variables to merge.
        """
        # Resolve command path
        executable = shutil.which(command) or command
        if not executable:
            raise ValueError(f"Command '{command}' not found.")

        # Merge environment variables
        current_env = os.environ.copy()
        if env:
            current_env.update(env)

        server_params = StdioServerParameters(
            command=executable,
            args=args,
            env=current_env
        )

        try:
            self._logger.info(f"Connecting to stdio server: {command} {args}")
            
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = transport
            
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self._timeout))
            )
            
            await self._session.initialize()
            self._logger.info("Stdio connection initialized successfully.")
            
        except Exception as e:
            self._logger.error(f"Failed to connect to stdio server: {e}")
            await self.cleanup()
            raise e

    async def connect_to_sse(self, url: str):
        """
        Connect to an MCP server using SSE (Server-Sent Events) transport.

        Args:
            url: SSE server endpoint URL.
        """
        try:
            self._logger.info(f"Connecting to SSE server: {url}")
            
            transport = await self._exit_stack.enter_async_context(
                sse_client(url=url)
            )
            read, write = transport
            
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self._timeout))
            )
            
            await self._session.initialize()
            self._logger.info("SSE connection initialized successfully.")
            
        except Exception as e:
            self._logger.error(f"Failed to connect to SSE server: {e}")
            await self.cleanup()
            raise e

    async def list_tools(self) -> List[Any]:
        """
        Retrieve the list of available tools from the server.

        Returns:
            List of tool definitions.
        """
        if not self._session:
            raise RuntimeError(f"Client {self._name} is not connected.")

        result = await self._session.list_tools()
        return result.tools if hasattr(result, 'tools') else []

    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Optional[Dict[str, Any]] = None,
        retries: int = 3, 
        delay: float = 1.0
    ) -> Any:
        """
        Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments as key-value pairs.
            retries: Maximum retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result as text, or error message if tool returned an error.
        """
        if not self._session:
            raise RuntimeError(f"Client {self._name} is not connected.")
        
        arguments = arguments or {}
        
        attempt = 0
        while attempt < retries:
            try:
                self._logger.debug(f"Calling tool '{tool_name}' (Attempt {attempt + 1}/{retries})")
                
                result = await self._session.call_tool(tool_name, arguments)
                
                # Extract text content from response
                text_content = "".join([
                    content.text 
                    for content in result.content 
                    if content.type == "text"
                ])

                # Handle tool-level errors (e.g., invalid query)
                # Return error as string to allow LLM self-correction
                if result.isError:
                    self._logger.warning(f"Tool '{tool_name}' returned an error: {text_content}")
                    return f"Tool Execution Error: {text_content}"

                return text_content

            except Exception as e:
                attempt += 1
                self._logger.warning(f"Tool execution failed: {e}. Retrying in {delay}s...")
                if attempt >= retries:
                    self._logger.error(f"Max retries reached for tool '{tool_name}'.")
                    raise e
                await asyncio.sleep(delay)

    async def cleanup(self):
        """
        Release resources and close connections.
        """
        async with self._cleanup_lock:
            if self._session:
                self._logger.info("Cleaning up MCP client resources...")
                try:
                    await self._exit_stack.aclose()
                    self._session = None
                except Exception as e:
                    self._logger.error(f"Error during cleanup: {e}")