"""Logging callback handler for LangChain agents."""

import logging

from langchain_core.callbacks import BaseCallbackHandler


class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs agent activity."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logger.info(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        self.logger.info(f"LLM ended with response: {response}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.logger.info(f"Tool started with input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        self.logger.info(f"Tool ended with output: {output}")

    def on_text(self, text, **kwargs):
        self.logger.info(f"Agent generated text: {text}")
