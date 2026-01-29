"""
    An MCP server for executing read-only SQL on PostgreSQL
"""
import os
import logging
import json
from typing import Any, List, Dict

import asyncpg
import click
import pandas as pd
from mcp.server.fastmcp import FastMCP
import re


# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DB_DSN = os.environ.get("DB_DSN")
TARGET_SCHEMA = os.environ.get("DB_SCHEMA", "public")
DEFAULT_TIMEOUT = float(os.environ.get("POSTGRES_TIMEOUT"))

async def get_db_connection():
    """Establish a connection to the PostgreSQL database"""
    try:
        conn = await asyncpg.connect(DB_DSN)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to DB: {e}")
        return None

def format_results(records: List[asyncpg.Record]) -> str:
    """
    Format query results for LLM consumption.
    Returns list of dictionaries for structured data representation.
    """
    if not records:
        return "Query executed successfully. No results returned."

    # Convert records to dict format
    data = [dict(r) for r in records]

    # Optional: Use Markdown tables for small result sets (uncomment if needed)
    # if len(data) < 50:
    #     try:
    #         df = pd.DataFrame(data)
    #         return df.to_markdown(index=False)
    #     except ImportError:
    #         return json.dumps(data, indent=2, default=str)

    # Return compact JSON format
    return data

def build_server(port: int) -> FastMCP:
    """
    Initializes the Postgres MCP server.

    Args:
        port: Port for SSE.

    Return: The MCP server.
    """
    mcp = FastMCP("postgres_executor", port=port)

    @mcp.tool()
    async def run_read_only_sql(query: str) -> str:
        """
        Execute a read-only SQL query on the PostgreSQL database.
        
        Use this tool to inspect data, check table schemas, or analyze statistics.
        
        Args:
            query: The SQL query string to execute
        """
        # Optional: Keyword-based filtering (uncomment if needed)
        # Note: True security relies on database user permissions, not keyword filtering
        # forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "truncate", "grant", "revoke"]
        # if any(keyword in query.lower() for keyword in forbidden_keywords):
        #     logger.warning(f"Blocked potential write operation: {query}")
        #     return "Error: This tool is strictly for READ-ONLY operations. INSERT, UPDATE, DELETE, etc., are not allowed."

        conn = await get_db_connection()
        if not conn:
            return "System Error: Could not connect to the database."

        try:
            # Enforce read-only transaction to prevent any write operations
            await conn.execute("BEGIN READ ONLY;")

            await conn.execute(f"SET search_path TO {TARGET_SCHEMA}")

            statements = [s.strip() + ";" for s in re.split(r';', query) if s.strip()]
            formatted_results = []
            for s_id, s in enumerate(statements):
                logger.info(f"Executing SQL: {s}")
                # Set query timeout to prevent long-running queries from blocking
                result = await conn.fetch(s, timeout=DEFAULT_TIMEOUT)
                formatted_result = format_results(result)
                formatted_results.append(
                    {
                        "sql_id": s_id,
                        "result": formatted_result
                    }
                )
                
            return json.dumps(formatted_results, ensure_ascii=False, default=str)

        # except asyncpg.UndefinedTableError as e:
        #     return f"SQL Error: Table not found. Please use 'list_tables' to verify table names. Original error: {e}"
        # except asyncpg.UndefinedColumnError as e:
        #     return f"SQL Error: Column not found. Please use 'describe_table' to verify column names. Original error: {e}"
        except asyncpg.InsufficientPrivilegeError:
            return "Database Error: Permission denied. You do not have access to this data."
        except asyncpg.PostgresError as e:
            return f"SQL Execution Error: {str(e)}"
        except Exception as e:
            return f"Unexpected Error: {str(e)}"
        finally:
            await conn.close()

    @mcp.tool()
    async def list_tables() -> str:
        """
        List all available tables in the database schema.
        """
        conn = await get_db_connection()
        if not conn:
            return "System Error: Could not connect to the database."
        
        try:
            # Query tables in the specified schema
            query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = $1
            ORDER BY table_name;
            """
            results = await conn.fetch(query, TARGET_SCHEMA)
            
            if not results:
                return f"No tables found in schema '{TARGET_SCHEMA}'. Please verify the DB_SCHEMA configuration."
                
            return json.dumps(format_results(results), ensure_ascii=False)
        except Exception as e:
            return f"Error listing tables: {str(e)}"
        finally:
            await conn.close()

    @mcp.tool()
    async def describe_tables(table_names: List[str]) -> str:
        """
        Get the schema definition for MULTIPLE tables at once in Compact JSON format.
        
        Args:
            table_names: A list of table names to describe.
        """
        conn = await get_db_connection()
        if not conn:
            return "System Error: Could not connect to the database."

        try:
            # Batch query for table schemas
            query = """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = ANY($1) 
            AND table_schema = $2
            ORDER BY table_name, ordinal_position;
            """
            
            results = await conn.fetch(query, table_names, TARGET_SCHEMA)
            
            if not results:
                return json.dumps({
                    "error": f"No tables found matching: {table_names} in schema '{TARGET_SCHEMA}'"
                })

            # Build compact JSON structure
            # Format: { "table_name": [ ["col1", "type", "nullable"], ["col2", "type", "nullable"] ] }
            tables_data = {}

            for row in results:
                t_name = row['table_name']
                if t_name not in tables_data:
                    tables_data[t_name] = []

                # Store values in order matching format_definition below
                tables_data[t_name].append([
                    row['column_name'],
                    row['data_type'],
                    "YES" if row['is_nullable'] == "YES" else "NO"
                ])

            # Add format_definition to clarify array structure for LLM
            final_output = {
                "format_definition": ["column_name", "data_type", "is_nullable"],
                "tables": tables_data
            }

            return json.dumps(final_output, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": f"Error describing tables: {str(e)}"})
        finally:
            await conn.close()
            
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
    Starts the Postgres MCP server.

    Args:
        port: Port for SSE.
        transport: The transport type, e.g., `stdio` or `sse`.
    """
    logger.info("Starting the Postgres MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())