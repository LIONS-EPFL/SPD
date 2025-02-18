# Adapted from submission file of JailbreakBench 
# https://github.com/JailbreakBench/jailbreakbench/blob/main/src/jailbreakbench/submission.py

import os
from collections.abc import Callable
from typing import Any, Literal

import torch
import numpy as np

from SPD.config import API_KEYS, Model
from jailbreakbench.llm.llm_wrapper import LLM
 
from SPD.litellm_spd import LLMLiteLLM_SPD

def calculate_logits(
    save_path: str,
    prompts_dict: dict[str, dict[str, str]],
    llm_provider: Literal["vllm"] | Literal["litellm"] = "litellm",
    defense: str | None = None,
):
    """Query the final responses for each prompt and calculate the logit values.

    Args:
        prompts_dict: A nested dictionary, model_name -> behavior -> prompt
        llm_provider: The LLM provider to use, either "vllm" or "litellm"
    Returns:
        A nested dictionary, model_name -> behavior -> (prompts, responses, classifications)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)


    llm_constructor: Callable[[str], LLM]

    if llm_provider == "vllm":
        try:
            from src.vllm_spd import LLMvLLM_SPD
        except ImportError:
            raise ImportError("vllm is not installed. Install it with `pip install vllm`.")
        llm_constructor = lambda model_name: LLMvLLM_SPD(model_name=model_name)
    elif llm_provider == "litellm":
        llm_constructor = lambda model_name: LLMLiteLLM_SPD(
            model_name=model_name, api_key=os.environ[API_KEYS[Model(model_name)]]
        )
    else:
        raise ValueError(f"Invalid llm_provider: {llm_provider}. Should be one of 'vllm' or 'litellm'.")

    for model_name, behaviors_dict in prompts_dict.items():
        llm = llm_constructor(model_name=model_name)
        logits =  generate_with_logits(
                    model_name=model_name, llm=llm, behavior_prompt_dict=behaviors_dict, defense=defense
                )

        file_path = os.path.join(save_path, f"{model_name}.npy")
        np.save(file_path, logits)

        # Delete previous LLM; otherwise, we risk going OOM
        # when using vLLM since multiple models will be loaded
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return 

def generate_with_logits(
    model_name: str, llm: LLM, behavior_prompt_dict: dict[str, str], defense: str | None
):
    none_behaviors = [behavior for behavior, prompt in behavior_prompt_dict.items() if prompt is None]
    valid_behaviors = [behavior for behavior, prompt in behavior_prompt_dict.items() if prompt is not None]
    valid_prompts = [behavior_prompt_dict[beh] for beh in valid_behaviors]

    print(f"Querying responses for {model_name}.")
    logits = llm._get_logits(valid_prompts, max_new_tokens=None, defense=defense)
    return logits