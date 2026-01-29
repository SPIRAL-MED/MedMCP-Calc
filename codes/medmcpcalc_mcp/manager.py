import asyncio
import json
import logging
import os
import re

from rich import print
from typing import Dict, List, Any, Optional, Union
from contextlib import AsyncExitStack
from .client import MCPClient



class MCPManager:
    """
    Manages MCP server configurations and client connections.
    Supports both Stdio and SSE transport protocols.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize MCPManager.
        
        Args:
            config_path: Path to JSON configuration file.
        """
        self._server_configs: Dict[str, Dict] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        
        if config_path:
            self.load_server_config(config_path)
            
        self._client_pool: Dict[str, MCPClient] = {}
        self._initialized = False
       
    def load_server_config(self, config_path: str):
        """
        Load server configurations from a JSON file.
        
        Supports two formats:
            1. {"mcpServers": {"name": {...}}} - Standard MCP format
            2. {"name": {...}} - Flat format
        
        Args:
            config_path: Path to the configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                configs = data.get("mcpServers", data)
                
                for name, conf in configs.items():
                    self.add_server_config(name, conf)
            except json.JSONDecodeError as e:
                self._logger.error(f"Failed to parse config file: {e}")
                raise

    def add_server_config(self, server_name: str, config: Dict[str, Any]):
        """
        Add or update a server configuration.
        
        Args:
            server_name: Unique identifier for the server.
            config: Server configuration dictionary.
        """
        self._server_configs[server_name] = config
        self._logger.info(f"Added server config: {server_name}")

    def get_server_config(self, server_name: str) -> Dict[str, Any]:
        """
        Retrieve configuration for a specific server.
        
        Args:
            server_name: Name of the server.
        
        Returns:
            Server configuration dictionary.
        """
        if server_name not in self._server_configs:
            raise ValueError(f"Server '{server_name}' not found.")
        return self._server_configs[server_name]
    
    
    
    async def initialize(self):
        """
        Initialize all MCP client connections concurrently.
        """
        if self._initialized:
            return
            
        self._logger.info("Initializing all MCP client connections...")
        
        tasks = []
        for server_name in self._server_configs.keys():
            tasks.append(self._build_and_store_client(server_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            server_name = list(self._server_configs.keys())[i]
            if isinstance(result, Exception):
                self._logger.error(f"Failed to initialize client for {server_name}: {result}")
            else:
                self._logger.info(f"Successfully initialized client for {server_name}")
        
        self._initialized = True
        
        
    async def _build_and_store_client(self, server_name: str, transport: str = "stdio"):
        """
        Build a client and store it in the connection pool.
        
        Args:
            server_name: Name of the server.
            transport: Transport protocol ('stdio' or 'sse').
        """
        try:
            client = await self.build_client(server_name, transport)
            self._client_pool[server_name] = client
            return True
        except Exception as e:
            self._logger.error(f"Failed to build client for {server_name}: {e}")
            raise e
     

    async def build_client(self, server_name: str, transport: str = "stdio", timeout: int = 60) -> MCPClient:
        """
        Build and connect an MCP client for a specific server.
        
        Args:
            server_name: Name of the server.
            transport: Transport protocol ('stdio' or 'sse').
            timeout: Connection timeout in seconds.
        
        Returns:
            Connected MCPClient instance.
        """
        config = self.get_server_config(server_name)
        client = MCPClient(name=f"{server_name}_client", timeout=timeout)

        try:
            if transport == "stdio":
                command = config.get("command")
                args = config.get("args", [])
                env = config.get("env", {})

                if not command:
                    raise ValueError(f"Missing 'command' in config for server {server_name}")

                await client.connect_to_stdio_server(command=command, args=args, env=env)

            elif transport == "sse":
                url = config.get("url")
                if not url:
                     raise ValueError(f"Missing 'url' in config for server {server_name}")
                
                await client.connect_to_sse(url=url)

            else:
                raise ValueError(f"Unsupported transport: {transport}")

            return client

        except Exception as e:
            self._logger.error(f"Failed to build client for {server_name}: {e}")
            await client.cleanup()
            raise e
        
    def get_client(self, server_name: str) -> MCPClient:
        """
        Get a client from the connection pool.
        
        Args:
            server_name: Name of the server.
        
        Returns:
            MCPClient instance.
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")
            
        if server_name not in self._client_pool:
            raise ValueError(f"No client found for server: {server_name}")
            
        return self._client_pool[server_name]

    
    async def cleanup(self):
        """
        Clean up all client connections and reset the manager state.
        """
        self._logger.info("Cleaning up all client connections...")
        
        cleanup_tasks = []
        for server_name, client in self._client_pool.items():
            cleanup_tasks.append(self._cleanup_client(server_name, client))
            
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._client_pool.clear()
        self._initialized = False
        
    async def _cleanup_client(self, server_name: str, client: MCPClient):
        """
        Clean up a single client connection.
        
        Args:
            server_name: Name of the server.
            client: MCPClient instance to clean up.
        """
        try:
            await client.cleanup()
            self._logger.info(f"Successfully cleaned up client for {server_name}")
        except Exception as e:
            self._logger.error(f"Failed to cleanup client for {server_name}: {e}")
            
            
    async def execute(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        transport: str = "stdio"
    ) -> Any:
        """
        Execute a tool on a specific server using an existing connection.
        
        Args:
            server_name: Name of the server.
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            transport: Transport protocol (kept for API compatibility).
        
        Returns:
            Tool execution result.
        """
        client = self.get_client(server_name)
        
        try:
            result = await client.call_tool(tool_name, arguments or {})
            return result
        except Exception as e:
            self._logger.error(f"Failed to execute {tool_name} on {server_name}: {e}")
            raise

    
    async def list_tools(
        self, 
        servers: Union[str, List[str], None] = None
    ) -> Dict[str, Union[List[Any], Exception]]:
        """
        Fetch tools from specified server(s).
        
        Args:
            servers: Server name(s) to query. If None, queries all configured servers.
            
        Returns:
            Dict mapping server names to their tool lists or exceptions.
        """
        
        # Normalize servers to list
        if servers is None:
            servers = list(self._server_configs.keys())
        elif isinstance(servers, str):
            servers = [servers]
        else:
            pass

        results = {}
        
        async def _fetch_one(server_name):
            """Fetch tools from a single server."""
            try:
                client = self.get_client(server_name)
                tools = await client.list_tools()
                return server_name, tools
            except Exception as e:
                return server_name, e

        # Fetch concurrently
        tasks = [_fetch_one(server) for server in servers]
        pairs = await asyncio.gather(*tasks)
        
        for server_name, result in pairs:
            results[server_name] = result
            
        return results
        
    
    async def list_tools_str(
        self,
        server_names: Union[str, List[str], None] = None
    ) -> str:
        """
        Fetch tools and return as JSON string.
        
        Args:
            server_names: Server name(s) to query. If None, queries all.
            
        Returns:
            JSON string of tools with server_name injected, or error message.
        """
        
        try:
            tools = await self.list_tools(server_names)
            tools_data = []
            
            for server_name, tool_list in tools.items():
                for t in tool_list:
                    # Convert to dict (handles both Pydantic models and regular objects)
                    t_dict = t.model_dump() if hasattr(t, 'model_dump') else t.__dict__.copy()
                    # Inject server name for identification
                    t_dict['server_name'] = server_name 
                    tools_data.append(t_dict)

            return json.dumps(tools_data, ensure_ascii=False, indent=2)
        except Exception:
            return "No tools available or failed to fetch tools."
