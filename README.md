<div align="center">

#  PyVision: Agentic Vision with Dynamic Tooling



<a href="https://arxiv.org/abs/2506.08011" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-PyVision-red?logo=arxiv" height="20" />
</a>
<a href="https://agent-x.space/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Homepage-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/spaces/Agents-X/PyVision-fully-isolated" target="_blank">
    <img alt="HF Model: ViGaL" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Demo-PyVision-ffc107?color=ffc107&logoColor=white" height="20" />
</a>


</div>


## 🎯Overview
Multi-modal large language model (MLLM) tool using with pre-defined toolset could limit the application scenarios. To dynamically generate and utilize adaptive tools in diverse scenarios, we propose PyVision, an interactive and multi-turn framework capable of dynamic tool generation with Python code, designed for multi-modal reasoning.
In particular, we implement this system with three core designs: (1) cross-turn persistent Python runtime, (2) file system-isolated I/O and (3) a comprehensive system prompt. When applying PyVision on diverse domains, we found that it could generate unexpected tools, e.g., PyVision calculate and visualize the image pixel histogram to help count the specific patterns in an image.

## 🚩News
- [2025-7-8] 🚀🚀🚀 We are excited to release `PyVision`, inluding:
  - [Techniqual report](), code and [online demo](https://huggingface.co/spaces/Agents-X/PyVision-fully-isolated).

## 📋Contents
- [Intallation](#installation)
- [Run](#run)
- [Citation](#citation)

## 📦Installation
Prepare the running environment, both for the main process and the *environment runtime*.
```bash
git clone https://github.com/agents-x-project/PyVision.git
cd PyVision

conda create -n pyvision python=3.10
conda activate pyvision
pip install -r requirements.txt
```

## 💥Run PyVision

### 1. Setup API Config
Before running `PyVision`, you need to first setup the API config file, including the key and the base_url. We provide three types of clients: OpenAI, Azure and vLLM.

#### OpenAI Client
```bash
# ./api_config_files/api_config_openai.json
{
    "api_key": [
        "sk-xxx"
    ],
    "base_url": "xxx"
}
```
#### Azure Client
```bash
# ./api_config_files/api_config_azure.json
{
    "api_key": [
        "sk-xxx"
    ],
    "base_url": "xxx"
}
```
#### vLLM Client 
```bash
# ./api_config_files/api_config_vllm.json
{
    "api_key": [
        "sk-xxx"
    ],
    "base_url": "xxx"
}
```
### 2. Run 
If you have setup the OpenAI API config file, you can run the `run.sh` file.
```bash
# openai client

python main.py \
    --image_path ./test_data/one_image_demo.png \
    --question "From the information on that advertising board, what is the type of this shop?" \
    --api_config ./api_config_files/api_config_openai.json \
    --client_type openai \
    --prompt_template ./prompt_template/prompt_template_vis.json \
    --prompt vistool_with_img_info_v2 \
    --exe_code \
    --max_tokens 10000 \
    --temperature 0.6 \
    --output_dir ./test_data \
    --save_messages 
```


### 3. Visualization
After running the **run.sh** file, the generated message is stored at `./test_data/test_message.json`. <br>
Upload the message file to our hosted visualization HuggingFace space: [visualization demo](https://huggingface.co/spaces/Agents-X/data-view).

## 📜Citation
```bibtex
@article{zhao2025pyvision,
  title={PyVision: Agentic Vision with Dynamic Tooling.},
  author={Zhao, Shitian and zhang, Haoquan and Lin, Shaoheng and Li, Ming and Wu, Qilong and Zhang, Kaipeng and Wei, Chen},
  journal={arxiv preprint},
  year={2025},
}
```