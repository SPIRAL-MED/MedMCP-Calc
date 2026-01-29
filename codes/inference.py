#!/usr/bin/env python3
"""
MCP-Agent Benchmark Runner
"""

import os
import argparse
import asyncio
import logging
import jsonlines

from tqdm import tqdm
from rich import print

from medmcpcalc_mcp.manager import MCPManager
from agent.llm import MyOpenAILLM
from agent.react import ReAct




# =============================================================================
# Configuration
# =============================================================================
class Args:
    """
    Configuration class to handle command-line arguments and system settings.
    """
    def parseargs(self):
        parser = argparse.ArgumentParser()
        
        # --- Data configuration ---
        parser.add_argument('--input_path', type=str, default="../benchmark/tasks.jsonl", help="Path to the input JSONL file containing tasks.")
        parser.add_argument("--l", type=int, default=0, help="Left index for data slicing (inclusive).")
        parser.add_argument("--r", type=int, default=None, help="Right index for data slicing (exclusive), None means no limit.")
        parser.add_argument("--max_workers", type=int, default=2, help="Number of concurrent threads for processing.")
        
        # --- llm model ---
        parser.add_argument('--llm_api_test', action='store_true', help="Flag to run LLM API connectivity test.")
        parser.add_argument("--think_switch", type=int, default=0, 
                            help="Thinking mode control for hybrid reasoning models: 0 = standard mode (regular models), 1 = enable thinking (hybrid models), -1 = disable thinking (hybrid models)")
        
        parser.add_argument('--model_name', type=str, default="claude-opus-4-5", help="Name of the LLM model to use.")
        parser.add_argument("--base_url", type=str, default="")
        parser.add_argument("--api_key", type=str, default="", help="API key for authentication.")
        
        # --- Agent ---
        parser.add_argument("--max_iterations", type=int, default=80, help="Maximum iterations for the ReAct agent loop.")
        parser.add_argument("--prompt_path", type=str, default="./agent/react_prompt.j2", help="Path to the Jinja2 prompt template.")
        
        # --- MCP ---
        parser.add_argument('--mcp_test', action='store_true', help="Flag to run MCP server connectivity test.")
        parser.add_argument("--config_path", type=str, default="./medmcpcalc_mcp/servers_config.json", help="Path to the MCP servers configuration file.")
        
        self.pargs = parser.parse_args()
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)

    def __init__(self) -> None:
        self.parseargs()
        
        os.makedirs("../outputs", exist_ok=True)
        if self.think_switch == 0:
            self.output_path = f"../outputs/{self.model_name}_results.jsonl"
        elif self.think_switch == 1:
            self.output_path = f"../outputs/{self.model_name}_think_results.jsonl"
        elif self.think_switch == -1:
            self.output_path = f"../outputs/{self.model_name}_nonthink_results.jsonl"
            
        os.makedirs("./logs", exist_ok=True)
        self.log_path = f"./logs/run_{self.model_name}.log"

args = Args()  



# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(levelname)s: %(message)s',
    filename=args.log_path,
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



# =============================================================================
# Task Execution
# =============================================================================
async def run_single_task(
    data_item: dict, 
    args: Args, 
    mcp: "MCPManager", 
    llm: "MyOpenAILLM", 
    semaphore: asyncio.Semaphore
) -> dict:
    """
    Execute a single benchmark task with concurrency control.
    
    Initializes a ReAct agent, processes the medical assessment question,
    and returns structured results. Handles exceptions gracefully to ensure
    partial results are captured even on failure.
    
    Args:
        data_item: Dictionary containing task data with keys:
            - task_id: Unique identifier for the task
            - patient_id: Patient identifier for the assessment
            - fuzzy_question: The medical question to answer ...
        args: Application configuration.
        mcp: Initialized MCP manager for tool access.
        llm: Initialized LLM client.
        semaphore: Asyncio semaphore for concurrency limiting.
        
    Returns:
        dict: Result dictionary containing:
            - task_id: The task identifier
            - process_success: 1 if successful, 0 if failed
            - final_answer: The agent's final answer
            - history: ReAct history string
            - history_list: List of agent ReAct outputs
            - failed_response: Any failed API responses
    """
    async with semaphore:
        try:
            # Initialize the ReAct agent
            agent = ReAct(mcp, llm, args.max_iterations, args.prompt_path)

            # Construct the question with patient context
            question = (
                f"I am assessing the patient with patient_id '{data_item['patient_id']}'. "
                f"Please explicitly identify and use applicable medical calculators to "
                f"quantify the assessment. Show the calculated results.\n\n"
                f"{data_item['fuzzy_question']}"
            )
            
            # Run the agent
            final_answer = await agent.run(question)

            # Return the results
            result = {
                "task_id": data_item["task_id"],
                "process_success": 1,
                "final_answer": final_answer,
                "history": agent.get_history(),
                "history_list": agent.agent_outputs,
                "failed_response": agent.failed_response
            }
            return result
            
        except Exception as e:
            logger.error(f"Error processing ID {data_item.get('task_id')}: {e}")
            
            result = {
                "task_id": data_item["task_id"],
                "process_success": 0,
                "final_answer": None,
                "history": agent.get_history(),
                "history_list": agent.agent_outputs,
                "failed_response": agent.failed_response
            }
            return result


async def main():
    # --- Data Loading ---
    with jsonlines.open(args.input_path, "r") as reader:
        data = [obj for obj in reader]
    print(f"Total data size: {len(data)}")
    
    # --- Resume Support: Filter Already Processed Tasks ---
    # Load task_ids that have already been processed to enable resume
    processed_ids = set()
    if os.path.exists(args.output_path):
        with jsonlines.open(args.output_path, "r") as reader:
            processed_ids = {obj["task_id"] for obj in reader}
            
   # Filter and slice the remaining data
    rest_data = [obj for obj in data if obj["task_id"] not in processed_ids]
    rest_data = rest_data[args.l: args.r]
    print(f"Remaining data size: {len(rest_data)}")

    
    # --- Initialize Shared Dependencies ---
    # MCP manager handles connections to all configured tool servers
    mcp = MCPManager(args.config_path)
    await mcp.initialize()
    
    # LLM client handles all model inference requests
    llm = MyOpenAILLM(args.model_name, args.base_url, args.api_key, args.think_switch)
    
    # --- Concurrency Control Setup ---
    # Semaphore limits concurrent executions to prevent API overload
    sem = asyncio.Semaphore(args.max_workers)
    
    # Create task coroutines (not yet scheduled)
    tasks = []
    for item in rest_data:
        task = run_single_task(item, args, mcp, llm, sem)
        tasks.append(task)

    # --- Execute with Progress Tracking and Incremental Writes ---
    print(f"Starting execution with {args.max_workers} parallel workers...")
    
    # Open output file in append mode for incremental writes
    with jsonlines.open(args.output_path, mode='a') as writer:
        # asyncio.as_completed yields futures as they complete (not in order)
        # tqdm wraps this to show progress
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await coro
            # Write immediately to prevent data loss on interruption
            writer.write(result)



# =============================================================================
# Testing Functions
# =============================================================================
async def mcp_test():
    """
    Integration test suite for MCP (Model Context Protocol) servers.
    
    Tests connectivity and basic functionality of all configured MCP servers:
    1. Google Search Server: Tests search query and URL fetch capabilities
    2. Python Executor Server: Tests code execution functionality
    3. PostgreSQL Server: Tests database connectivity and query execution
    
    This function is useful for:
    - Verifying server configurations before running benchmarks
    - Debugging connection issues
    - Validating that all required tools are available
    """
    print("\n Starting MCP Manager Integration Tests...\n")

    # --- Initialize MCP Manager ---
    try:
        manager = MCPManager(args.config_path)
        await manager.initialize()
    except Exception as e:
        print(f"❌ Configuration file loading failed: {e}")
        return
    
    # Display all available tools across all servers
    tools = await manager.list_tools()
    # print(tools)
    
    
    # =========================================================================
    # Test 1: Google Search Server
    # =========================================================================
    print("--------------------------------------------------")
    print("🔎 Testing Server: google_search")
    try:
        # Step A: Discover available tools
        print("   Fetching tool list...")
        tools = await manager.list_tools("google_search")
        tool_names = [t.name for t in tools["google_search"]]
        print(f"   ✅ Discovered tools: {tool_names}")

        # Step B: Test the 'search' tool
        if "search" in tool_names:
            print("   Executing tool 'search' (query='Dog')...")
            result = await manager.execute(
                server_name="google_search",
                tool_name="search",
                arguments={"query": "dog"}
            )
            result_str = str(result)
            print(f"   ✅ Execution successful! Result preview: {result_str[:500]}...")
        else:
            print("   ⚠️ Tool 'search' not found")

        # Step C: Test the 'fetch' tool
        if "fetch" in tool_names:
            print("   Executing tool 'fetch' (url='https://www.mdcalc.com/calc/3836/...')...")
            result = await manager.execute(
                server_name="google_search",
                tool_name="fetch",
                arguments={"url": "https://www.mdcalc.com/calc/3836/fisher-grading-scale-subarachnoid-hemorrhage-sah"}
            )
            result_str = str(result)
            print(f"   ✅ Execution successful! Result preview: {result_str[:500]}...")
        else:
            print("   ⚠️ Tool 'fetch' not found")
    except Exception as e:
        print(f"   ❌ Google Search test failed: {e}")


    # =========================================================================
    # Test 2: Python Executor Server
    # =========================================================================
    print("\n--------------------------------------------------")
    print("🐍 Testing Server: python_executor")
    try:
        # Step A: Discover available tools
        print("   Fetching tool list...")
        tools = await manager.list_tools("python_executor")
        tool_names = [t.name for t in tools["python_executor"]]
        print(f"   ✅ Discovered tools: {tool_names}")

        # Step B: Test the 'run_python' tool with a simple calculation
        if "run_python" in tool_names:
            code_snippet = "import math; print(f'Sqrt of 144 is {math.sqrt(144)}')"
            print(f"   Executing tool 'run_python'...")
            result = await manager.execute(
                server_name="python_executor",
                tool_name="run_python",
                arguments={"code": code_snippet}
            )
            print(f"   ✅ Execution result:\n{result}")
        else:
            print("   ⚠️ Tool 'run_python' not found")
    except Exception as e:
        print(f"   ❌ Python Executor test failed: {e}")


    # =========================================================================
    # Test 3: PostgreSQL Executor Server
    # =========================================================================
    print("\n--------------------------------------------------")
    print("🐘 Testing Server: postgres_executor")
    try:
        # Step A: Discover available tools
        print("   Fetching tool list...")
        tools = await manager.list_tools("postgres_executor")
        tool_names = [t.name for t in tools["postgres_executor"]]
        print(f"   ✅ Discovered tools: {tool_names}")
        
        # Step B: Test the 'run_read_only_sql' tool
        # Note: Requires a running PostgreSQL instance with proper configuration
        if "run_read_only_sql" in tool_names:
            # Example queries (commented out alternatives for different test scenarios):
            print("   Executing tool 'run_read_only_sql' ...")
            sql_query = "SELECT * FROM laboratory_result WHERE patient_id = 'P004_5e9098ea' LIMIT 1;"
            print(f"   Executing SQL: {sql_query}")
            
            result = await manager.execute(
                server_name="postgres_executor",
                tool_name="run_read_only_sql",
                arguments={"query": sql_query}
            )
            print(f"   ✅ Execution result:\n{result}")
        else:
            print("   ⚠️ Tool 'run_read_only_sql' not found")

        # Step C: Test the 'list_tables' tool for schema discovery
        if "list_tables" in tool_names:
            print("   Executing tool 'list_tables' ...")
            result = await manager.execute(
                server_name="postgres_executor",
                tool_name="list_tables",
                arguments={}
            )
            print(f"   ✅ Execution result:\n{result}")
        else:
            print("   ⚠️ Tool 'list_tables' not found")
    except Exception as e:
        print(f"   ❌ PostgreSQL test failed (database may not be connected): {e}")
    
    print("\n🏁 Testing complete.")

    

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    # Entry point: dispatch to appropriate handler based on command-line flags
    
    if args.llm_api_test:
        # Run LLM API connectivity test
        async def llm_api_test():
            """Quick test to verify LLM API endpoint is reachable and functional."""
            llm = MyOpenAILLM(args.model_name, args.base_url, args.api_key, args.think_switch)
            await llm.test_api()
        asyncio.run(llm_api_test())
        
    elif args.mcp_test:
        # Run MCP server integration tests
        asyncio.run(mcp_test())
        
    else:
        # Run the main benchmark
        asyncio.run(main())