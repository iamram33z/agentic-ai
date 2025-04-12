# src/fintech_ai_bot/agents/base.py
# (Using Toolkit type hint)

from abc import ABC, abstractmethod
from typing import List, Optional
from agno.tools.toolkit import Toolkit # Use Toolkit
from agno.agent import Agent as AgnoAgent
from agno.models.base import Model
from agno.models.groq import Groq
from agno.run.response import RunResponse
from fintech_ai_bot.config import AgentModelConfig # Import only needed config part
from fintech_ai_bot.utils import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """
    Abstract Base Class for all specialized AI agents within the FinTech bot.
    """

    def __init__(self, model_config: AgentModelConfig, tools: Optional[List[Toolkit]] = None):
        """
        Initializes the BaseAgent.

        Args:
            model_config: Configuration object for the language model (id, temp, etc.).
            tools: An optional list of Agno Toolkit instances the agent can use.
        """
        self.name: str = self.get_agent_name()
        self.role: str = self.get_agent_role()
        self.instructions: str = self.get_agent_instructions()
        self.tools: List[Toolkit] = tools or []
        self.model: Model = self._create_model(model_config)
        self._agno_agent: AgnoAgent = self._create_agno_agent()
        logger.info(f"Initialized agent: '{self.name}' with model '{model_config.id}'")

    @staticmethod
    @abstractmethod
    def get_agent_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_agent_role() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_agent_instructions() -> str:
        pass

    def _create_model(self, model_config: AgentModelConfig) -> Model:
        """Creates the language model instance based on the provided configuration."""
        try:
            if not model_config.id:
                raise ValueError("Model ID is missing in agent configuration.")
            model_instance = Groq(
                id=model_config.id,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens
            )
            logger.debug(f"Model '{model_config.id}' created for agent '{self.name}'.")
            return model_instance
        except Exception as e:
            logger.critical(f"Failed to create LLM model '{model_config.id}' for agent '{self.name}': {e}", exc_info=True)
            raise RuntimeError(f"Model creation failed for agent '{self.name}'") from e

    def _create_agno_agent(self) -> AgnoAgent:
        """Creates the underlying Agno library agent instance."""
        try:
            agno_instance = AgnoAgent(
                name=self.name,
                role=self.role,
                model=self.model,
                tools=self.tools, # Expects List[Toolkit]
                instructions=self.instructions,
                markdown=True # Defaulting to True
            )
            logger.debug(f"Agno agent instance created for '{self.name}'.")
            return agno_instance
        except Exception as e:
            logger.critical(f"Failed to create Agno agent instance for '{self.name}': {e}", exc_info=True)
            raise RuntimeError(f"Agno agent instance creation failed for '{self.name}'") from e

    def run(self, prompt: str) -> Optional[str]:
        """Executes the agent with the given prompt."""
        if not prompt:
            logger.warning(f"Agent '{self.name}' received an empty prompt. Skipping run.")
            return None
        logger.debug(f"Running agent '{self.name}' with prompt: '{prompt[:150]}...'")
        try:
            response_obj: RunResponse = self._agno_agent.run(prompt)
            if isinstance(response_obj, RunResponse) and response_obj.content:
                content = str(response_obj.content).strip()
                logger.debug(f"Agent '{self.name}' finished successfully. Response length: {len(content)}")
                return content
            else:
                logger.warning(f"Agent '{self.name}' returned unexpected response type ({type(response_obj)}) or empty content.")
                if hasattr(response_obj, 'error') and response_obj.error:
                    logger.error(f"Agent '{self.name}' run resulted in error state: {response_obj.error}")
                return None
        except Exception as e:
            logger.error(f"Agent '{self.name}' failed during run execution: {e}", exc_info=True)
            raise