"""
LLM client manager with automatic Anthropic → OpenAI fallback.

The LLMManager class abstracts all LLM interactions, providing:
  - Structured output via Pydantic models
  - Automatic fallback if the primary provider fails
  - Consistent retry behaviour
  - A single place to swap models or providers
"""

from __future__ import annotations

import logging
from typing import List, Optional, Type, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config import AppConfig, get_config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMManager:
    """
    Manages LLM clients with automatic fallback from Anthropic to OpenAI.

    Usage:
        manager = LLMManager()

        # Plain invocation
        response = await manager.ainvoke(messages)

        # Structured output (returns a Pydantic model instance)
        plan = await manager.ainvoke_structured(messages, ResearchPlan)
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or get_config()
        self._primary: Optional[BaseChatModel] = None
        self._fallback: Optional[BaseChatModel] = None

    # ------------------------------------------------------------------ #
    # Client Accessors
    # ------------------------------------------------------------------ #

    @property
    def primary(self) -> BaseChatModel:
        """Lazily initialise and return the primary Anthropic client."""
        if self._primary is None:
            self._primary = ChatAnthropic(
                model=self._config.anthropic_model,
                api_key=self._config.anthropic_api_key,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
        return self._primary

    @property
    def fallback(self) -> Optional[BaseChatModel]:
        """Lazily initialise and return the OpenAI fallback client (if configured)."""
        if self._fallback is None and self._config.has_openai_fallback:
            self._fallback = ChatOpenAI(
                model=self._config.openai_model,
                api_key=self._config.openai_api_key,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
        return self._fallback

    # ------------------------------------------------------------------ #
    # Invocation Helpers
    # ------------------------------------------------------------------ #

    async def ainvoke(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """
        Invoke the primary LLM, falling back to OpenAI on failure.

        Args:
            messages: List of LangChain BaseMessage objects.

        Returns:
            The model's text response as a string.

        Raises:
            RuntimeError: If both primary and fallback LLMs fail.
        """
        primary_error = None
        try:
            response = await self.primary.ainvoke(messages)
            return response.content
        except Exception as primary_err:
            primary_error = primary_err
            logger.warning(
                "Primary LLM (%s) failed: %s. Trying fallback...",
                self._config.anthropic_model,
                primary_error,
            )

        if self.fallback is None:
            raise RuntimeError(
                f"Primary LLM failed and no fallback is configured. "
                f"Set OPENAI_API_KEY in your .env file. "
                f"Primary error: {primary_error}"
            )

        try:
            response = await self.fallback.ainvoke(messages)
            logger.info("Fallback LLM (%s) succeeded.", self._config.openai_model)
            return response.content
        except Exception as fallback_err:
            raise RuntimeError(
                f"Both LLMs failed. "
                f"Primary ({self._config.anthropic_model}): {primary_error}. "
                f"Fallback ({self._config.openai_model}): {fallback_err}"
            ) from fallback_err

    async def ainvoke_structured(
        self,
        messages: List[BaseMessage],
        output_schema: Type[T],
    ) -> T:
        """
        Invoke the LLM and parse the response into a Pydantic model.

        Uses LangChain's .with_structured_output() which leverages tool-calling
        under the hood for reliable JSON extraction.

        Args:
            messages: List of LangChain BaseMessage objects.
            output_schema: The Pydantic model class to parse into.

        Returns:
            An instance of output_schema populated from the LLM response.

        Raises:
            RuntimeError: If both primary and fallback LLMs fail.
        """
        primary_error = None
        try:
            structured_llm = self.primary.with_structured_output(output_schema)
            result = await structured_llm.ainvoke(messages)
            if not isinstance(result, output_schema):
                raise TypeError(f"Expected {output_schema.__name__}, got {type(result)}")
            return result
        except Exception as primary_err:
            primary_error = primary_err
            logger.warning(
                "Primary LLM structured output failed: %s. Trying fallback...",
                primary_error,
            )

        if self.fallback is None:
            raise RuntimeError(
                f"Primary LLM structured output failed and no fallback is configured. "
                f"Error: {primary_error}"
            )

        try:
            structured_llm = self.fallback.with_structured_output(output_schema)
            result = await structured_llm.ainvoke(messages)
            logger.info("Fallback LLM structured output succeeded.")
            return result
        except Exception as fallback_err:
            raise RuntimeError(
                f"Both LLMs failed for structured output. "
                f"Primary: {primary_error}. Fallback: {fallback_err}"
            ) from fallback_err

    def get_model_info(self) -> dict:
        """Return a dict with current model configuration for logging/display."""
        return {
            "primary_model": self._config.anthropic_model,
            "fallback_model": self._config.openai_model
            if self._config.has_openai_fallback
            else "none",
            "temperature": self._config.llm_temperature,
            "max_tokens": self._config.llm_max_tokens,
        }


# Module-level singleton accessor
_manager: Optional[LLMManager] = None


def get_llm_manager(config: Optional[AppConfig] = None) -> LLMManager:
    """
    Return (or create) the module-level LLMManager singleton.

    Args:
        config: Optional config override; uses get_config() if not provided.

    Returns:
        The singleton LLMManager instance.
    """
    global _manager
    if _manager is None:
        _manager = LLMManager(config)
    return _manager
