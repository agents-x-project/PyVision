# vis_inference_demo_gpt.py

import sys
# sys.path.append("/mnt/petrelfs/zhaoshitian/TIR-Data-Synthesis")
import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
import argparse
from inference_engine.safe_persis_shared_vis_python_exe import PythonExecutor, ImageRuntime
from openai import OpenAI
import anthropic

def encode_image(image):
    """
    将PIL.Image对象或图像文件路径转换为base64编码字符串，并获取分辨率信息
    
    参数:
        image: 可以是PIL.Image对象或图像文件路径
        
    返回:
        包含以下键的字典:
        - 'base64': base64编码的字符串
        - 'width': 图片宽度(像素)
        - 'height': 图片高度(像素)
        - 'resolution': 字符串形式的"宽度x高度"
    """
    img_obj = None
    
    if isinstance(image, str):
        # 处理文件路径的情况
        img_obj = Image.open(image)
        with open(image, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # 处理PIL.Image对象的情况
        img_obj = image
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 获取分辨率信息
    width, height = img_obj.size
    
    return {
        'base64': base64_str,
        'width': width,
        'height': height
    }

def encode_image_with_resize(image):
    """
    将PIL.Image对象或图像文件路径转换为base64编码字符串，并获取分辨率信息。
    如果分辨率大于1024x1024，则缩小为原来的一半。
    
    参数:
        image: 可以是PIL.Image对象或图像文件路径
        
    返回:
        包含以下键的字典:
        - 'base64': base64编码的字符串
        - 'width': 图片宽度(像素)
        - 'height': 图片高度(像素)
        - 'resolution': 字符串形式的"宽度x高度"
    """
    img_obj = None
    
    if isinstance(image, str):
        img_obj = Image.open(image)
    else:
        img_obj = image

    # 检查尺寸是否大于1024x1024，如果是则resize为一半
    width, height = img_obj.size
    if width > 1024 or height > 1024:
        new_size = (width // 2, height // 2)
        img_obj = img_obj.resize(new_size, Image.LANCZOS)
        width, height = img_obj.size

    # 转base64
    buffered = BytesIO()
    img_obj.save(buffered, format='PNG')
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        'base64': base64_str,
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}"
    }

def check(evaluator, pred_ans, real_ans):
    if len(pred_ans) == 0:
        return []
    correctness = evaluator.score(pred_ans, real_ans)
    return correctness

def excute_codes(codes, messages, executor: PythonExecutor):
    no_code_idx = []
    codes_use = []
    for i, code in enumerate(codes):
        if code == "":
            no_code_idx.append(i)
        else:
            codes_use.append(code)
    batch_results = executor.batch_apply(codes_use, messages)
    return batch_results, no_code_idx

def process_prompt_init(question, image_path_list, prompt_template, prompt_type, api_name):
    with open(prompt_template, "r") as fin:
        sys = json.load(fin)
    prompt_prefix = sys[prompt_type]

    image_path = image_path_list[0]

    if "<IMAGE_PLACE_HOLDER_0>" in question:
        if "no_tool" in prompt_type:

            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            question_with_options = question
            question = prompt_prefix.format(query=question_with_options)

            parts = question.split("<IMAGE_PLACE_HOLDER_0>")
            content = []
            
            # 添加图片前的文本（如果有）
            if parts[0].strip():
                content.append({"type": "text", "text": parts[0].strip()})
            # 添加图片
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            
            # 添加图片后的文本（如果有）
            if len(parts) > 1 and parts[1].strip():
                content.append({"type": "text", "text": parts[1].strip()})

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            return messages

        else:
            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            width = img_result['width']
            height = img_result['height']
            question_with_options = question
            question = prompt_prefix.format(query=question_with_options, width=str(width), height=str(height))

            # 将问题分割成图片前后的部分
            parts = question.split("<IMAGE_PLACE_HOLDER_0>")
            # 构建带有image_clue标签的消息
            content = []
            
            # 添加图片前的文本（如果有）
            if parts[0].strip():
                content.append({"type": "text", "text": parts[0].strip()})
            
            # 添加带标签的图片
            content.append({"type": "text", "text": "<image_clue_0>"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            content.append({"type": "text", "text": "</image_clue_0>\n\n"})
            
            # 添加图片后的文本（如果有）
            if len(parts) > 1 and parts[1].strip():
                content.append({"type": "text", "text": parts[1].strip()})

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]

            return messages

    else:
        if "no_tool" in prompt_type:

            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            question_with_options = question

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": prompt_prefix.format(query=question_with_options)}]
                }
            ]

            return messages

        else:
            if "claude" in api_name:
                img_result = encode_image_with_resize(image_path)
            else:
                img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            width = img_result['width']
            height = img_result['height']
            question_with_options = question

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "<image_clue_0>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": "</image_clue_0>\n\n"}] + [{"type": "text", "text": prompt_prefix.format(query=question_with_options, width=str(width), height=str(height))}]
                }
            ]

            return messages

def process_prompt_init_multi_images(question, image_path_list, prompt_template, prompt_type, api_name):
    with open(prompt_template, "r") as fin:
        sys = json.load(fin)
    prompt_prefix = sys[prompt_type]
    
    # 准备图片数据
    image_data = []
    image_information = ""
    
    for i, image_path in enumerate(image_path_list):
        if "claude" in api_name:
            img_result = encode_image_with_resize(image_path)
        else:
            img_result = encode_image(image_path)
        image_base64 = img_result['base64']
        width = img_result['width']
        height = img_result['height']
        
        image_data.append({
            "index": i,
            "base64": image_base64,
            "width": width,
            "height": height,
            "placeholder": f"<IMAGE_PLACE_HOLDER_{i}>"
        })
        
        image_information += f"width of image_clue_{i}: {width}, height of image_clue_{i}: {height}\n"
    
    # 格式化问题
    formatted_question = prompt_prefix.format(query=question, image_information=image_information)
    
    # 检查是否有图片占位符
    has_placeholders = any(f"<IMAGE_PLACE_HOLDER_{i}>" in formatted_question for i in range(len(image_path_list)))
    
    if has_placeholders:
        # 如果有占位符，按占位符位置插入图片
        if "no_tool" in prompt_type:
            # 初始化内容列表
            content = []
            remaining_text = formatted_question
            
            # 处理每个占位符
            for img_data in image_data:
                placeholder = img_data["placeholder"]
                if placeholder in remaining_text:
                    # 分割文本
                    parts = remaining_text.split(placeholder, 1)
                    
                    # 添加占位符前的文本
                    if parts[0]:
                        content.append({"type": "text", "text": parts[0]})
                    
                    # 添加图片
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                    
                    # 更新剩余文本
                    remaining_text = parts[1]
            
            # 添加最后剩余的文本
            if remaining_text:
                content.append({"type": "text", "text": remaining_text})
            
            messages = [{"role": "user", "content": content}]
            return messages
        else:
            # 使用 image_clue 标签
            content = []
            remaining_text = formatted_question
            
            # 处理每个占位符
            for img_data in image_data:
                placeholder = img_data["placeholder"]
                if placeholder in remaining_text:
                    # 分割文本
                    parts = remaining_text.split(placeholder, 1)
                    
                    # 添加占位符前的文本
                    if parts[0]:
                        content.append({"type": "text", "text": parts[0]})
                    
                    # 添加带标签的图片
                    i = img_data["index"]
                    content.append({"type": "text", "text": f"<image_clue_{i}>"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                    content.append({"type": "text", "text": f"</image_clue_{i}>\n\n"})
                    
                    # 更新剩余文本
                    remaining_text = parts[1]
            
            # 添加最后剩余的文本
            if remaining_text:
                content.append({"type": "text", "text": remaining_text})
            
            messages = [{"role": "user", "content": content}]
            return messages
    else:
        # 如果没有占位符，按原来的方式处理
        if "no_tool" in prompt_type:
            content = []
            
            # 先添加所有图片
            for i, img_data in enumerate(image_data):
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
            
            # 然后添加文本
            content.append({"type": "text", "text": formatted_question})
            
            messages = [{"role": "user", "content": content}]
            return messages
        else:
            # 使用 image_clue 标签
            content = []
            
            # 添加所有图片，带标签
            for i, img_data in enumerate(image_data):
                content.append({"type": "text", "text": f"<image_clue_{i}>"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}})
                content.append({"type": "text", "text": f"</image_clue_{i}>\n\n"})
            
            # 添加文本
            content.append({"type": "text", "text": formatted_question})
            
            messages = [{"role": "user", "content": content}]
            return messages


def update_messages_with_excu_content(image_nums_in_input, messages, images_result, text_result, error_result, image_clue_idx):
    if error_result is None:
        new_messages = []
        image_content = []
        for message_item in messages[:-1]:
            new_messages.append(message_item)

        assistant_message_item = messages[-1]['content']
        interpreter_message_text_prefix = [{"type": "text", "text": f"<interpreter>\nText Result:\n{text_result}\nImage Result:\n"}]
        if images_result is not None:
            for image_base64_item in images_result[image_clue_idx-image_nums_in_input:]:
                interpreter_message_images = [{"type": "text", "text": f"<image_clue_{image_clue_idx}>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_item}"}}] + [{"type": "text", "text": f"</image_clue_{image_clue_idx}>"}]
                image_content += interpreter_message_images
                image_clue_idx += 1
        else:
            image_content = [{"type": "text", "text": "None"}]
        interpreter_message_text_profill = [{"type": "text", "text": "</interpreter>\n"}]

        interpreter_message_item = interpreter_message_text_prefix + image_content + interpreter_message_text_profill
        new_messages.append({"role": "assistant", "content": assistant_message_item})
        new_messages.append({"role": "user", "content": interpreter_message_item})
    else:
        new_messages = []
        for message_item in messages[:-1]:
            new_messages.append(message_item)
    
        assistant_message_item = messages[-1]['content']
        interpreter_message_text_prefix = [{"type": "text", "text": f"<interpreter>{error_result}"}]
        interpreter_message_text_profill = [{"type": "text", "text": "</interpreter>\n"}]
    
        interpreter_message_item = interpreter_message_text_prefix + interpreter_message_text_profill
        new_messages.append({"role": "assistant", "content": assistant_message_item})
        new_messages.append({"role": "user", "content": interpreter_message_item})

    return new_messages, image_clue_idx

def update_messages_with_code(messages, generated_content):
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}</code>\n"}]
    }

    messages.append(message_item)
    return messages

def update_messages_with_text(messages, generated_content):
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}"}]
    }

    messages.append(message_item)
    return messages

def call_chatgpt_api(args, messages, client, max_tokens=10000, stop=None, temperature=0.6):
    """Call ChatGPT API with the given messages"""
    try:
        client_type = args.client_type
        api_name = args.api_name
    except:
        client_type = args['client_type']
        api_name = args['api_name']
    
    if client_type == "openai" or client_type == "azure":
        response = client.chat.completions.create(
            model=api_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=stop,
            timeout=300
        )
        response_text = response.choices[0].message.content
    elif client_type == "anthropic":
        message = client.messages.create(
            model=api_name,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
            top_p=1.0,
            stop_sequences=stop
        )  
        response_text = message.content[0].text if isinstance(message.content, list) else message.content
    elif client_type == "vllm":
        response = client.chat.completions.create(
            model=api_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=stop
        )
        response_text = response.choices[0].message.content
    else:
        print("Your args.client_type must be one of openai, azure, anthropic and vllm.")
        return None, None
    
    # 检查是否遇到停止标记
    stop_reason = None
    if stop and any(s in response_text for s in stop):
        for s in stop:
            if s in response_text:
                stop_reason = s
                break
    else:
        if client_type in ["openai", "azure", "vllm"]:
            stop_reason = response.choices[0].finish_reason
        else:
            stop_reason = "stop"

    if "<code>" in response_text:
        stop_reason = "</code>"
    
    return response_text, stop_reason

def evaluate_single_data(args, data, client, executor):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
        temperature = args.temperature
        api_name = args.api_name
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']
        temperature = args['temperature']
        api_name = args['api_name']

    image_path_list = data['image_path_list']

    if "no_tool" in prompt:
        if len(image_path_list) == 1:
            messages = process_prompt_init(data["question"], image_path_list, prompt_template, prompt, api_name)
        elif len(image_path_list) >= 2:
            messages = process_prompt_init_multi_images(data["question"], image_path_list, prompt_template, prompt, api_name)
    else:
        if len(image_path_list) == 1:
            prompt = "vistool_with_img_info_v2"
            messages = process_prompt_init(data["question"], image_path_list, prompt_template, prompt, api_name)
        elif len(image_path_list) >= 2:
            prompt = "vistool_with_img_info_multi_image"
            messages = process_prompt_init_multi_images(data["question"], image_path_list, prompt_template, prompt, api_name)
    
    # 生成初始响应
    response_text, pred_stop_reason = call_chatgpt_api(
        args,
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None,
        temperature=temperature
    )
    
    # 处理响应
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = len(image_path_list)
    
    while True:
        # 检查是否需要执行代码
        if exe_code and pred_stop_reason == "</code>":
            # 提取要执行的代码
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # 执行代码
            exe_result = excute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_excu_content(len(image_path_list), messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # 生成下一部分响应
            response_text, pred_stop_reason = call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None,
                temperature=temperature
            )

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response


def evaluate_single_data_multi_images(args, data, client, executor):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']

    messages = process_prompt_init_multi_images(data["question"], data['image_path_list'], prompt_template, prompt)
    
    # 生成初始响应
    response_text, pred_stop_reason = call_chatgpt_api(
        args,
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None
    )
    
    # 处理响应
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = data['image_nums_in_input']
    
    while True:
        # 检查是否需要执行代码
        if exe_code and pred_stop_reason == "</code>":
            # 提取要执行的代码
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # 执行代码
            exe_result = excute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_excu_content(data['image_nums_in_input'], messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # 生成下一部分响应
            response_text, pred_stop_reason = call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None
            )

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response

def evaluate_single_data_video(args, data, client, executor):
    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']

    messages = process_prompt_init_multi_images(data["question"], data['image_path_list'], prompt_template, prompt)
    
    # 生成初始响应
    response_text, pred_stop_reason = call_chatgpt_api(
        args,
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None
    )
    
    # 处理响应
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = data['image_nums_in_input']
    
    while True:
        # 检查是否需要执行代码
        if exe_code and pred_stop_reason == "</code>":
            # 提取要执行的代码
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # 执行代码
            exe_result = excute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_excu_content(data['image_nums_in_input'], messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # 生成下一部分响应
            response_text, pred_stop_reason = call_chatgpt_api(
                args,
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None
            )

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response


# New wrapper functions for safe execution with cleanup
def evaluate_batch_with_cleanup(args, data_list, client):
    """Wrapper function to ensure proper cleanup of resources when processing multiple items"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data(args, data, client, executor)
                results.append(result)
            except Exception as e:
                print(f"Error processing data item: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor

def evaluate_single_with_cleanup(args, data, client):
    """Wrapper function for evaluating a single item with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)

    try:
        result = evaluate_single_data(args, data, client, executor)
        return result
    finally:
        # Ensure cleanup of persistent worker
        del executor

def evaluate_multi_images_with_cleanup(args, data_list, client):
    """Wrapper function for multi-image evaluation with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data_multi_images(args, data, client, executor)
                results.append(result)
            except Exception as e:
                print(f"Error processing multi-image data: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor

def evaluate_video_with_cleanup(args, data_list, client):
    """Wrapper function for video evaluation with proper cleanup"""
    # Initialize executor with process isolation
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for data in data_list:
            try:
                result = evaluate_single_data_video(args, data, client, executor)
                results.append(result)
            except Exception as e:
                print(f"Error processing video data: {str(e)}")
                results.append((None, f"Error: {str(e)}"))
                # Reset the executor for the next item
                executor.reset()
        
        return results
    finally:
        # Ensure cleanup of persistent worker
        del executor


# Main execution functions
def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='Visual Inference Demo with GPT')
    
    # Model arguments
    parser.add_argument('--api_name', type=str, default='gpt-4-vision-preview',
                        help='API model name')
    parser.add_argument('--client_type', type=str, choices=['openai', 'azure', 'anthropic', 'vllm'],
                        default='openai', help='Type of client to use')
    
    # Prompt arguments
    parser.add_argument('--prompt_template', type=str, required=True,
                        help='Path to prompt template JSON file')
    parser.add_argument('--prompt', type=str, default='vistool_with_img_info_v2',
                        help='Prompt type to use from template')
    
    # Execution arguments
    parser.add_argument('--exe_code', action='store_true',
                        help='Whether to execute code blocks')
    parser.add_argument('--max_tokens', type=int, default=10000,
                        help='Maximum tokens for API response')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for API generation')
    
    # Input/Output arguments
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input JSON file with questions and images')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save results')
    
    # API keys (can also be set via environment variables)
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (defaults to environment variable)')
    parser.add_argument('--api_base', type=str, default=None,
                        help='API base URL for Azure or custom endpoints')
    
    args = parser.parse_args()
    
    # Initialize client
    if args.client_type == 'openai':
        client = OpenAI(api_key=args.api_key or os.getenv('OPENAI_API_KEY'))
    elif args.client_type == 'azure':
        client = OpenAI(
            api_key=args.api_key or os.getenv('AZURE_API_KEY'),
            api_base=args.api_base or os.getenv('AZURE_API_BASE'),
            api_type='azure',
            api_version='2023-05-15'
        )
    elif args.client_type == 'anthropic':
        client = anthropic.Anthropic(api_key=args.api_key or os.getenv('ANTHROPIC_API_KEY'))
    elif args.client_type == 'vllm':
        client = OpenAI(
            api_key=args.api_key or "EMPTY",
            base_url=args.api_base or "http://localhost:8000/v1"
        )
    else:
        raise ValueError(f"Unsupported client type: {args.client_type}")
    
    # Load input data
    with open(args.input_file, 'r') as f:
        data_list = json.load(f)
    
    # Process data
    results = []
    
    # Determine data type and use appropriate evaluation function
    if isinstance(data_list, list) and len(data_list) > 0:
        sample_data = data_list[0]
        
        if 'image_nums_in_input' in sample_data:
            # Multi-image or video data
            if 'video' in args.input_file.lower():
                results = evaluate_video_with_cleanup(args, data_list, client)
            else:
                results = evaluate_multi_images_with_cleanup(args, data_list, client)
        else:
            # Single image data
            results = evaluate_batch_with_cleanup(args, data_list, client)
    else:
        print("Invalid input data format")
        return
    
    # Save results
    output_data = []
    for i, (data, (messages, response)) in enumerate(zip(data_list, results)):
        output_item = {
            'id': data.get('id', i),
            'question': data['question'],
            'response': response,
            'messages': messages,
            'image_paths': data.get('image_path_list', [])
        }
        
        # Add ground truth if available
        if 'answer' in data:
            output_item['ground_truth'] = data['answer']
        
        output_data.append(output_item)
    
    # Write output
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_file}")


def run_single_example(question, image_paths, args_dict, client=None):
    """
    Convenience function to run a single example programmatically
    
    Args:
        question: The question to ask
        image_paths: List of image file paths
        args_dict: Dictionary containing configuration (prompt_template, prompt, exe_code, etc.)
        client: Pre-initialized client (optional)
    
    Returns:
        tuple: (messages, response)
    """
    # Convert args_dict to namespace for compatibility
    class Args:
        pass
    
    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)
    
    # Initialize client if not provided
    if client is None:
        if args.client_type == 'openai':
            client = OpenAI(api_key=args.api_key or os.getenv('OPENAI_API_KEY'))
        elif args.client_type == 'anthropic':
            client = anthropic.Anthropic(api_key=args.api_key or os.getenv('ANTHROPIC_API_KEY'))
        else:
            raise ValueError(f"Unsupported client type: {args.client_type}")
    
    # Prepare data
    data = {
        'question': question,
        'image_path_list': image_paths if isinstance(image_paths, list) else [image_paths]
    }
    
    # Run evaluation
    return evaluate_single_with_cleanup(args, data, client)


# Example usage function
def example_usage():
    """Example of how to use the inference functions"""
    
    # Configuration
    args_dict = {
        'prompt_template': 'path/to/prompt_template.json',
        'prompt': 'vistool_with_img_info_v2',
        'exe_code': True,
        'max_tokens': 10000,
        'temperature': 0.6,
        'api_name': 'gpt-4-vision-preview',
        'client_type': 'openai',
        'api_key': None  # Will use environment variable
    }
    
    # Single image example
    question = "What objects can you see in this image?"
    image_path = "path/to/image.jpg"
    
    messages, response = run_single_example(question, image_path, args_dict)
    print(f"Response: {response}")
    
    # Multi-image example
    question_multi = "What are the differences between these images?"
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    
    messages_multi, response_multi = run_single_example(question_multi, image_paths, args_dict)
    print(f"Multi-image response: {response_multi}")


if __name__ == "__main__":
    # Check if running as script with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Run example if no arguments provided
        print("No arguments provided. Running example usage...")
        print("For command-line usage, run with --help flag")
        # example_usage()  # Uncomment to run example