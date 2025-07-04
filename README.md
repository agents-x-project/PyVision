<div align="center">

#  PyVision: Agentic Vision with Dynamic Tooling



<a href="https://arxiv.org/abs/2506.08011" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-PyVision-red?logo=arxiv" height="18" />
</a>
<a href="https://agent-x.space/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_PyVision-blue.svg" height="18" />
</a>
<a href="https://huggingface.co/spaces/Agents-X/PyVision-fully-isolated" target="_blank">
    <img alt="HF Model: ViGaL" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Demo-PyVision-ffc107?color=ffc107&logoColor=white" height="18" />
</a>


</div>


## Overview
Multi-modal large language model (MLLM) tool using with pre-defined toolset could limit the application scenarios. To dynamically generate and utilize adaptive tools in diverse scenarios, we propose PyVision, an interactive and multi-turn framework capable of dynamic tool generation with Python code, designed for multi-modal reasoning.
In particular, we implement this system with three core designs: (1) cross-turn persistent Python runtime, (2) file system-isolated I/O and (3) a comprehensive system prompt. When applying PyVision on diverse domains, we found that it could generate unexpected tools, e.g., PyVision calculate and visualize the image pixel histogram to help count the specific patterns in an image.

## Contents
- [Intallation](#installation)
- [Run](#run)
- [Citation](#citation)

## Installation
Prepare the running environment, both for the main process and the *environment runtime*.
```bash
git clone https://github.com/agents-x-project/PyVision.git
cd PyVision

conda create -n pyvision python=3.10
conda activate pyvision
pip install -r requirements.txt
```
Instantiate the API config file.
```bash
# api_config.json
{
    "api_key": [
        "sk-xxx"
    ],
    "base_url": "xxx"
}
```

## Run
```bash
python main.py
```

## Citation
```bibtex
@article{zhao2025pyvision,
  title={PyVision: Agentic Vision with Dynamic Tooling.},
  author={Zhao, Shitian and zhang, Haoquan and Lin, Shaoheng and Li, Ming and Wu, Qilong and Zhang, Kaipeng and Wei, Chen},
  journal={arxiv preprint},
  year={2025},
}
```