import logging
import openai
import re
import json
import time
from openai import AsyncOpenAI
from rich import print
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseLLM:
    """Base class for LLM API interactions with async support and retry logic."""
    
    def __init__(self, model_name, base_url, api_key, think_switch=0) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        
        self.think_switch = think_switch
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
    
    async def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM with JSON validation.
        
        Retries up to 5 times if JSON parsing fails.
        
        Args:
            prompt: Input prompt string.
            
        Returns:
            Raw response string (with think tags stripped).
        """
        json_retries = 0
        while json_retries < 5:
            messages = [{"role": "user", "content": prompt}]
            
            try:
                response = await self._call_api_with_retry(messages)
            except Exception as e:
                # Catches exceptions after all retries exhausted or non-retryable errors
                logging.error(f"[BaseLLM] Failed after retries: {e}")
                return f" [BaseLLM] Error: {e}"
            
            ans = response.choices[0].message.content
            if "<think>" in ans and "</think>" in ans:
                ans = re.sub('<think>.*</think>', '', ans, flags=re.DOTALL).strip()
            elif "</think>" in ans:
                ans = re.sub('.*</think>', '', ans, flags=re.DOTALL).strip()
            # Although JSON format was specified, some models still generate extra text.
            # To evaluate LLM performance without being hindered by formatting issues,
            # we apply stricter filtering here.
            # Filter directly based on JSON opening, since "thought" is a required field.
            elif re.search(r'\{\s*"thought":', ans):
                match = re.search(r'\{\s*"thought":', ans)
                # Truncate from the matched position (start of {)
                ans = ans[match.start():]
            
            try:
                cleaned = ans.strip().replace("```json", "").replace("```", "").strip()
                cleaned = json.loads(cleaned)
                
                return ans
            except Exception as e:
                json_retries += 1
                logging.info(
                    f"[BaseLLM] Failed When Jsonized {cleaned}",
                    f"attempt #{json_retries}..."
                )
            
        logging.info(f"[BaseLLM] Failed When Jsonized after attempting 5 times. Return the wrong test:\n{ans}")
        return ans
        
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(8),
        retry=retry_if_exception_type((
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError
        )),
        before_sleep=lambda retry_state: logging.info(
            f"[BaseLLM] Retrying API call due to {retry_state.outcome.exception()}, "
            f"attempt #{retry_state.attempt_number}..."
        ),
        reraise=True 
    )
    async def _call_api_with_retry(self, messages: list):
        """
        Make API call with automatic retry on transient failures.
        
        Retries up to 8 times with exponential backoff (1-30s).
        Handles: InternalServerError, APITimeoutError, APIConnectionError, RateLimitError.
        
        Args:
            messages: List of message dicts for chat completion.
            
        Returns:
            API response object.
        """
        if self.think_switch == 0:
            ans = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
        elif self.think_switch == 1:
            ans = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            )
        elif self.think_switch == -1:
            ans = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
        
        return ans
    
    async def close(self):
        """Close the AsyncOpenAI client connection."""
        await self.client.close()

    async def test_api(self) -> str:
        """Quick connectivity and response test."""
        print(f"\n--- Testing {self.__class__.__name__} Model {self.model_name}---")

        start_time = time.perf_counter()

        messages = [{"role": "user", "content": "Hello, introduce yourself briefly."}]
        try:
            response = await self._call_api_with_retry(messages)
            res = response.choices[0].message.content
        except Exception as e:
            logging.error(f"[BaseLLM] Test API failed: {e}")
            res = f"[BaseLLM] Error: {e}"

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f"Time taken: {elapsed_time:.4f} seconds")
        print(res)
        


class MyOpenAILLM(BaseLLM):
    """OpenAI-compatible LLM wrapper."""
    pass