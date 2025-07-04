import json
import sys
import os
import argparse
from openai import OpenAI
from inference_engine.vis_inference_demo_gpt import evaluate_single_data
from inference_engine.shared_vis_python_exe import PythonExecutor

image_path = "./test_data/one_image_demo.png"
prompt_template = "./prompt_template/prompt_template_vis.json"
prompt = "vistool_with_img_info_v2"
question = "From the information on that advertising board, what is the type of this shop?"

# init
api_config_path = "./api_config.json"
api_config = json.load(open(api_config_path, "r"))
api_key = api_config['api_key'][0]
base_url = api_config['base_url']
client = OpenAI(api_key=api_key, base_url=base_url)
executor = PythonExecutor()

# data preparation
data = {
    "question": question,
    "image_path_list": [image_path],
}

args = {
    "max_tokens": 10000,
    "prompt_template": prompt_template,
    "prompt": prompt,
    "exe_code": True,
    "temperature": 0.6,
    "client_type": "openai",
    "api_name": "gpt-4.1"
}

# inference
messages, final_response = evaluate_single_data(args, data, client, executor)

save_messages_path = "./test_data/test_messages.json"

with open(save_messages_path, "w", encoding="utf-8") as f:
    json.dump(messages, f, indent=4, ensure_ascii=False)