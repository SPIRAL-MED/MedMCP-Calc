"""
An MCP server for executing Python code safely via subprocess.
"""
import sys
import os
import asyncio
import click
import logging
from mcp.server.fastmcp import FastMCP


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_TIMEOUT = float(os.environ.get("PYTHON_TIMEOUT"))


async def execute_python_code(code: str) -> str:
    """
    Execute Python code in a subprocess and capture the output.

    Args:
        code: Python code string to execute.

    Returns:
        Execution output or error message.
    """
    # Use same Python environment as the MCP server
    command = [sys.executable, "-c", code]

    try:
        # Execute code asynchronously in subprocess
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Wait for output with timeout to prevent infinite loops
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=DEFAULT_TIMEOUT)
        except asyncio.TimeoutError:
            process.kill()
            return "Error: Execution timed out (limit: 30s)."

        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()

        output = []
        if stdout_str:
            output.append(f"Output:\n{stdout_str}")
        if stderr_str:
            output.append(f"Errors/Warnings:\n{stderr_str}")
        
        if not output:
            return "Code executed successfully (no output)."

        return "\n\n".join(output)

    except Exception as e:
        return f"System Error executing code: {str(e)}"

def build_server(port: int) -> FastMCP:
    """
    Initializes the Python Executor MCP server.

    Args:
        port: Port for SSE.

    Returns: The MCP server.
    """
    mcp = FastMCP("python_executor", port=port)

    @mcp.tool()
    async def run_python(code: str) -> str:
        """
        Run a Python script and return the standard output.
        
        Use this tool to perform calculations, data processing, or logic tasks 
        that are difficult for the LLM to do directly.
        
        Args:
            code: The valid Python code to execute.
                  Example: "import math; print(math.sqrt(144))"
        """
        logger.info(f"Executing code snippet (length: {len(code)})")

        # Basic security filtering (consider stronger sandboxing for production)
        forbidden_keywords = ["subprocess", "os.system", "shutil.rmtree"]
        for keyword in forbidden_keywords:
            if keyword in code:
                return f"Security Error: Use of '{keyword}' is restricted."

        result = await execute_python_code(code)
        return result

    # @mcp.tool()
    # async def list_installed_packages() -> str:
    #     """
    #     List all installed Python packages in the current environment.
    #     """
    #     try:
    #         # Run pip list
    #         process = await asyncio.create_subprocess_exec(
    #             sys.executable, "-m", "pip", "list",
    #             stdout=asyncio.subprocess.PIPE,
    #             stderr=asyncio.subprocess.PIPE
    #         )
    #         stdout, _ = await process.communicate()
    #         return stdout.decode()
    #     except Exception as e:
    #         return f"Error listing packages: {str(e)}"

    return mcp

@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option("--port", default="8000", help="Port to listen on for SSE")
def main(transport: str, port: str):
    """
    Starts the Python Executor MCP server.

    Args:
        port: Port for SSE.
        transport: The transport type, e.g., `stdio` or `sse`.
    """
    assert transport.lower() in ["stdio", "sse"], \
        "Transport should be `stdio` or `sse`"
    
    logger.info("Starting the Python Executor MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())