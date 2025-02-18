# Adapted from vllm file of JailbreakBench 
# https://github.com/JailbreakBench/jailbreakbench/blob/main/src/jailbreakbench/llm/vllm.py

import vllm
import numpy as np
from transformers import AutoTokenizer
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from SPD.config import CHAT_TEMPLATES, HF_MODEL_NAMES
from SPD.llm_wrapper_spd import LLM


class LLMvLLM_SPD(LLM):
    def __init__(self, model_name: str):
        """Initializes the LLMvLLM with the specified model name."""
        super().__init__(model_name)
        self.hf_model_name = HF_MODEL_NAMES[self.model_name]
        destroy_model_parallel()
        self.model = vllm.LLM(model=self.hf_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        # Manually set the chat template is not a default one from HF
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = CHAT_TEMPLATES[self.model_name]
        if self.temperature > 0:
            self.sampling_params = vllm.SamplingParams(
                temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_n_tokens
            )
        else:
            self.sampling_params = vllm.SamplingParams(temperature=0, max_tokens=self.max_n_tokens)

    def update_max_new_tokens(self, n: int | None):
        """Update the maximum number of generation tokens."""

        if n is not None:
            self.sampling_params.max_tokens = n

    def query_llm_logit(self, inputs: list[list[dict[str, str]]]):
        """Query the LLM with supplied inputs."""

        def convert_query_to_string(query) -> str:
            """Convert query into prompt string."""
            return self.tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)

        self.sampling_params.top_k = -1 
        self.sampling_params.top_p = 1
        self.sampling_params.logprobs = 32000

        prompts = [convert_query_to_string(query) for query in inputs]

        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        logits = [[list(log.values()) for log in output.outputs[0].logprobs] for output in outputs]

        for l in range(len(logits)):
            if len(logits[l]) < self.sampling_params.max_tokens:
                while (len(logits[l])) < self.sampling_params.max_tokens:
                    print(len(logits[l]))
                    logits[l].append([0 for _ in range(self.sampling_params.logprobs)])

        return np.array(logits)