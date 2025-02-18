# Adapted from vllm and llm_wrapper file of JailbreakBench 
# https://github.com/JailbreakBench/jailbreakbench/blob/main/src/jailbreakbench/llm/llm_wrapper.py

import abc
import os
import threading

from SPD.config import MAX_GENERATION_LENGTH, SYSTEM_PROMPTS, TEMPERATURE, TOP_P, Model

class LLM(abc.ABC):
    temperature: float
    top_p: float
    max_n_tokens: int

    def __init__(self, model_name: str):
        """Initializes the LLM with the specified model name."""
        try:
            self.model_name = Model(model_name.lower())
        except ValueError:
            raise ValueError(
                f"Invalid model name: {model_name}. Expected one of: {', '.join([model.value for model in Model])}"
            )
        self.system_prompt = SYSTEM_PROMPTS[self.model_name]
        self.temperature = TEMPERATURE
        self.top_p = TOP_P
        self.max_n_tokens = MAX_GENERATION_LENGTH
        self.lock = threading.Lock()

    @abc.abstractmethod
    def update_max_new_tokens(self, n: int | None):
        """Update the maximum number of generation tokens.
        Subclass should override."""
        raise NotImplementedError()

    @abc.abstractmethod
    def query_llm_logit(self, inputs: list[list[dict[str, str]]]):
        """Query the LLM with supplied inputs.
        Subclass should override."""
        raise NotImplementedError()

    def _get_logits(
        self, input_prompts: list[str], max_new_tokens: int | None = None, defense: str | None = None
    ):
        """Generates responses from the model for the given prompts."""

        # TODO: consider inlining this into `query`.

        self.update_max_new_tokens(max_new_tokens)

        # Create list of conversations
        convs_list = [self.convert_prompt_to_conversation(p) for p in input_prompts]

        # Query the LLM
        logits = self.query_llm_logit(convs_list)

        return logits

    def convert_prompt_to_conversation(self, prompt: str) -> list[dict[str, str]]:
        """Create a conversation dictionary containing the LLM-specific
        system prompt and the user-supplied input prompt."""
        if self.system_prompt is None:
            return [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]