# Single-pass Detection of Jailbreaking Input in Large Language Models

This repository is the original implementation of the paper **"Single-pass Detection of Jailbreaking Input in Large Language Models"** TMLR 2025.

The codebase is built on [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) and extends its functionality to our jailbreaking defense method SPD. 

## Setup

### **Step 1: Install JailbreakBench**

Follow the installation instructions from the original JailbreakBench repository:

```bash
conda create -n spd python=3.10
cd jailbreakbench
pip install jailbreakbench[vllm]
```
### **Step 2: Install additional libraries**

```bash
cd ..
pip install -r requirements.txt
```
---
## Usage

1. In `data` folder, we share a huge successfull jailbreaking dataset we generated for 4 models from 3 different attack ([GCG](https://github.com/llm-attacks/llm-attacks), [AutoDAN](https://github.com/SheltonLiu-N/AutoDAN) and [PAIR](https://github.com/patrickrchao/JailbreakingLLMs)) with more than 4000 samples in total.
2. You can run the evaluate.ipynb to save and load logit values, train a classifier and test on your data.

---
## Cite as:

```bibtex
@article{
candogan2025singlepass,
title={Single-pass Detection of Jailbreaking Input in Large Language Models},
author={Leyla Naz Candogan and Yongtao Wu and Elias Abad Rocamora and Grigorios Chrysos and Volkan Cevher},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=42v6I5Ut9a},
note={}
}
```

If you use the dataset in your work, please also consider citing its constituent sources of the attacks:

```bibtex
@misc{zou2023universal,
  title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
  year={2023},
  eprint={2307.15043},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
@inproceedings{liu2023autodan,
    title={Generating Stealthy Jailbreak Prompts on Aligned Large Language Models},
    author={Xiaogeng Liu and Nan Xu and Muhao Chen and Chaowei Xiao},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024},
    url={https://openreview.net/forum?id=7Jwpw4qKkb}
}
@misc{chao2023jailbreaking,
      title={Jailbreaking Black Box Large Language Models in Twenty Queries}, 
      author={Patrick Chao and Alexander Robey and Edgar Dobriban and Hamed Hassani and George J. Pappas and Eric Wong},
      year={2023},
      eprint={2310.08419},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@inproceedings{tdc2023,
  title={TDC 2023 (LLM Edition): The Trojan Detection Challenge},
  author={Mantas Mazeika and Andy Zou and Norman Mu and Long Phan and Zifan Wang and Chunru Yu and Adam Khoja and Fengqing Jiang and Aidan O'Gara and Ellie Sakhaee and Zhen Xiang and Arezoo Rajabi and Dan Hendrycks and Radha Poovendran and Bo Li and David Forsyth},
  booktitle={NeurIPS Competition Track},
  year={2023}
}
```
