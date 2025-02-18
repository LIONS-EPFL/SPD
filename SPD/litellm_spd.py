# Adapted from litellm file of JailbreakBench 
# https://github.com/JailbreakBench/jailbreakbench/blob/main/src/jailbreakbench/llm/litellm.py

import litellm
from transformers import AutoTokenizer

from SPD.config import HF_MODEL_NAMES, LITELLM_CUSTOM_PROMPTS, LITELLM_MODEL_NAMES
from SPD.llm_wrapper_spd import LLM

litellm.drop_params = True

import numpy as np

class LLMLiteLLM_SPD(LLM):
    _N_RETRIES: int = 10
    _RETRY_STRATEGY: str = "exponential_backoff_retry"

    _litellm_model_name: str
    _api_key: str
    system_prompt: str

    def __init__(self, model_name: str, api_key: str):
        """Initializes the LLMLiteLLM with the specified model name.
        For Together AI models, api_key is the API key for the Together AI API.
        For OpenAI models, api_key is the API key for the OpenAI API.
        """

        super().__init__(model_name)
        self._api_key = api_key
        self._litellm_model_name = LITELLM_MODEL_NAMES[self.model_name]
        self.generation_params: dict[str, int | float] = {}
        if self.temperature == 0:
            self.generation_params["temperature"] = 0
        else:
            self.generation_params["top_p"] = 0.9
        self.generation_params["max_tokens"] = self.max_n_tokens
        self.generation_params["seed"] = 0
        if self.model_name in LITELLM_CUSTOM_PROMPTS:
            litellm.register_prompt_template(
                model=self._litellm_model_name, roles=LITELLM_CUSTOM_PROMPTS[self.model_name]
            )
        if self.model_name in HF_MODEL_NAMES:
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAMES[self.model_name])
        else:
            self.tokenizer = None

    def update_max_new_tokens(self, n: int | None):
        """Update the maximum number of generation tokens."""

        if n is not None:
            self.generation_params["max_tokens"] = n

    def query_llm_logit(self, inputs: list[list[dict[str, str]]]):
        """Query the LLM with supplied inputs."""

        self.generation_params["logprobs"] = True
        self.generation_params["top_logprobs"] = 5
        
        outputs = litellm.batch_completion(
            model=self._litellm_model_name,
            messages=inputs,
            api_key=self._api_key,
            num_retries=self._N_RETRIES,
            retry_strategy=self._RETRY_STRATEGY,
            **self.generation_params,  # type: ignore
        )
                
        logits = [[[x['logprob'] for x in y['top_logprobs']] for y in output["choices"][0]["logprobs"]['content']] for output in outputs]
        
        for l in range(len(logits)):
            while len(logits[l]) < self.generation_params["max_tokens"]:
                logits[l].append([0,0,0,0,0])  

        return np.array(logits)  