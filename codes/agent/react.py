import json
import logging
from typing import Optional, Union, Dict, List, Any
from dataclasses import dataclass, field

from rich import print
from jinja2 import Template

from medmcpcalc_mcp.manager import MCPManager
from agent.llm import BaseLLM


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReAct:
    """
    ReAct (Reasoning + Acting) Agent implementation.
    
    Combines reasoning with tool execution in an iterative loop
    until a final answer is produced or max iterations are reached.
    """
    def __init__(self, mcp_manager: MCPManager, llm: BaseLLM, max_iterations: int, prompt_path: str) -> None:
        """
        Initialize the ReAct agent.

        Args:
            mcp_manager: Manager for MCP tool servers.
            llm: Language model instance for generation.
            max_iterations: Maximum reasoning-action cycles allowed.
            prompt_path: Path to the Jinja2 prompt template file.
        """
        self._mcp_manager = mcp_manager
        self._llm = llm
        
        self._max_iterations = max_iterations
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        self._prompt_template = Template(template_content)
        
        self._history: List[str] = []
        self.agent_outputs: List[Dict] = []
        
        self.failed_response = None
        
    def _add_history(self, history_type: str, message: str):
        """
        Append an entry to the conversation history.

        Args:
            history_type: Entry type (e.g., "Thought", "Action", "Result").
            message: Content of the history entry.
        """
        formatted_type = history_type.title()
        self._history.append(f"{formatted_type}: {message}")

    def get_history(self) -> str:
        """Return formatted history string for prompt injection."""
        return "\n".join(self._history) if self._history else "EMPTY"

    def clear_history(self):
        """Reset conversation history."""
        self._history = []

    async def _build_prompt(self, question: str) -> str:
        """
        Render the prompt template with current context.

        Args:
            question: User's input question.

        Returns:
            Complete prompt string ready for LLM.
        """
        tools_desc = await self._mcp_manager.list_tools_str()
        history_str = self.get_history()

        return self._prompt_template.render(
            TOOLS_PROMPT=tools_desc,
            HISTORY=history_str,
            QUESTION=question,
            MAX_STEPS=self._max_iterations
        )
        
    async def run(self, question: str) -> str:
        """
        Execute the ReAct loop for a given question.

        Args:
            question: User's input question.

        Returns:
            Final answer string or timeout message.

        Raises:
            ValueError: If LLM response is not valid JSON.
        """
        self.clear_history()
        
        for i in range(self._max_iterations):
            logger.info(f"--- Iteration {i+1} ---")
            
            # Build and send prompt to LLM
            prompt = await self._build_prompt(question)
            # logger.info(f"\n--- prompt ---: {prompt}\n")
            # logger.info(f"\n--- history ---: {self.get_history()}\n")
            
            response_text = await self._llm.generate(prompt)
            logger.info(f"\n--- response_text ---: {response_text}\n")
            
            # Parse JSON response
            try:
                cleaned = response_text.strip().replace("```json", "").replace("```", "").strip()
                parsed_response = json.loads(cleaned)
            except json.JSONDecodeError as e:
                self.failed_response = response_text
                logger.error(f"ReAct Agent -- JSON Parse Error: {e}, Response: {response_text}")
                raise ValueError(f"Invalid JSON response from LLM: {response_text}")
            except Exception as e:
                self.failed_response = response_text
                logger.error(f"ReAct Agent -- Error: {e}, Response: {response_text}")
                raise e
            self.agent_outputs.append(parsed_response)
            
            # Process ReAct components
            try:
                # Record thought
                thought = parsed_response.get("thought", "No thought provided.")
                self._add_history("Step", f"{i+1}")
                self._add_history("Thought", thought)
                logger.info(f"\n--- ReAct Step {i+1} ---\n")
                logger.info(f"\n--- Thought ---: {thought}\n")
                
                # Case A: Final answer provided
                if "answer" in parsed_response:
                    answer = parsed_response["answer"]
                    self._add_history("Answer", answer)
                    logger.info(f"\n--- Answer ---: {answer}\n")
                    return answer
                
                # Case B: Tool action required
                elif "action" in parsed_response:
                    action = parsed_response["action"]
                    
                    action_log = f"Using tool `{action.get('tool')}` in server `{action.get('server')}` with args {action.get('arguments')}"
                    self._add_history("Action", action_log)
                    logger.info(f"\n--- Action ---: {action_log}\n")
                    
                    # Execute tool via MCP manager
                    try:
                        tool_result = await self._mcp_manager.execute(
                            server_name=action.get("server"),
                            tool_name=action.get("tool"),
                            arguments=action.get("arguments", {})
                        )
                        
                        result_str = str(tool_result)
                        self._add_history("Result", result_str)
                        self.agent_outputs[-1]["result"] = result_str
                        logger.info(f"\n--- Result ---: {result_str}\n")
                        
                    except Exception as tool_err:
                        error_msg = f"Tool execution error: {str(tool_err)}"
                        self._add_history("Result", error_msg)
                
                else:
                    raise ValueError("Response must contain either 'action' or 'answer'.")
                    
            except ValueError as e:
                # Log parsing errors for LLM self-correction in next iteration
                self._add_history("Error", f"Parsing error: {str(e)}. Please output valid JSON.")
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self._add_history("Error", f"System error: {str(e)}")

        return "Max iterations reached without a final answer."