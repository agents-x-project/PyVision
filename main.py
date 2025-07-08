import json
import sys
import os
import argparse
from openai import OpenAI
from inference_engine.vis_inference_demo_gpt import evaluate_single_data, evaluate_single_with_cleanup
from inference_engine.safe_persis_shared_vis_python_exe import PythonExecutor

def main():
    # Configuration
    image_path = "./test_data/one_image_demo.png"
    prompt_template = "./prompt_template/prompt_template_vis.json"
    prompt = "vistool_with_img_info_v2"
    question = "From the information on that advertising board, what is the type of this shop?"

    # API configuration
    api_config_path = "./api_config.json"
    api_config = json.load(open(api_config_path, "r"))
    api_key = api_config['api_key'][0]
    base_url = api_config['base_url']
    
    # Initialize client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Data preparation
    data = {
        "question": question,
        "image_path_list": [image_path],
    }

    # Arguments configuration
    args = {
        "max_tokens": 10000,
        "prompt_template": prompt_template,
        "prompt": prompt,
        "exe_code": True,
        "temperature": 0.6,
        "client_type": "openai",
        "api_name": "gpt-4.1"
    }

    # Method 1: Using the wrapper function (recommended for automatic cleanup)
    print("Running inference with safe execution...")
    messages, final_response = evaluate_single_with_cleanup(args, data, client)

    # Save results
    save_messages_path = "./test_data/test_messages.json"
    with open(save_messages_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)
    
    print(f"Final response: {final_response}")
    print(f"Messages saved to: {save_messages_path}")

    # Method 2: Manual executor management (if you need more control)
    # This shows how to manually manage the executor lifecycle
    """
    executor = PythonExecutor(use_process_isolation=True)
    try:
        messages, final_response = evaluate_single_data(args, data, client, executor)
        
        save_messages_path = "./test_data/test_messages.json"
        with open(save_messages_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
            
    finally:
        # Ensure cleanup
        del executor
    """

def main_with_args():
    """Main function with command-line arguments support"""
    parser = argparse.ArgumentParser(description='Visual Question Answering with Code Execution')
    
    # Input arguments
    parser.add_argument('--image_path', type=str, default="./test_data/one_image_demo.png",
                        help='Path to the input image')
    parser.add_argument('--question', type=str, 
                        default="From the information on that advertising board, what is the type of this shop?",
                        help='Question to ask about the image')
    
    # Configuration arguments
    parser.add_argument('--api_config', type=str, default="./api_config.json",
                        help='Path to API configuration file')
    parser.add_argument('--client_type', type=str, default="openai",
                        help='Client Type')
    parser.add_argument('--prompt_template', type=str, default="./prompt_template/prompt_template_vis.json",
                        help='Path to prompt template file')
    parser.add_argument('--prompt', type=str, default="vistool_with_img_info_v2",
                        help='Prompt type to use')
    
    # Execution arguments
    parser.add_argument('--exe_code', action='store_true', default=True,
                        help='Whether to execute code blocks')
    parser.add_argument('--max_tokens', type=int, default=10000,
                        help='Maximum tokens for response')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for generation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default="./test_data",
                        help='Directory to save output files')
    parser.add_argument('--save_messages', action='store_true', default=True,
                        help='Whether to save the message history')
    
    args = parser.parse_args()
    
    # Load API configuration
    with open(args.api_config, 'r') as f:
        api_config = json.load(f)
    
    api_key = api_config['api_key'][0]
    base_url = api_config.get('base_url', None)
    
    # Initialize client
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Prepare data
    data = {
        "question": args.question,
        "image_path_list": [args.image_path],
    }
    
    # Prepare arguments
    eval_args = {
        "max_tokens": args.max_tokens,
        "prompt_template": args.prompt_template,
        "prompt": args.prompt,
        "exe_code": args.exe_code,
        "temperature": args.temperature,
        "client_type": args.client_type,
        "api_name": api_config.get('model', 'gpt-4.1')
    }
    
    # Run inference with safe execution
    print(f"Processing image: {args.image_path}")
    print(f"Question: {args.question}")
    print("Running inference with safe execution...")
    
    messages, final_response = evaluate_single_with_cleanup(eval_args, data, client)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_messages:
        messages_path = os.path.join(args.output_dir, "messages.json")
        with open(messages_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
        print(f"Messages saved to: {messages_path}")
    
    # Save response
    response_path = os.path.join(args.output_dir, "response.txt")
    with open(response_path, "w", encoding="utf-8") as f:
        f.write(final_response)
    print(f"Response saved to: {response_path}")
    
    # Print response
    print("\n" + "="*50)
    print("Final Response:")
    print("="*50)
    print(final_response)
    print("="*50 + "\n")

def batch_process():
    """Example function for batch processing multiple images"""
    api_config_path = "./api_config.json"
    api_config = json.load(open(api_config_path, "r"))
    api_key = api_config['api_key'][0]
    base_url = api_config['base_url']
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Example batch data
    batch_data = [
        {
            "question": "What objects can you see in this image?",
            "image_path_list": ["./test_data/image1.png"],
        },
        {
            "question": "What is the main color in this image?",
            "image_path_list": ["./test_data/image2.png"],
        },
        # Add more items as needed
    ]
    
    args = {
        "max_tokens": 10000,
        "prompt_template": "./prompt_template/prompt_template_vis.json",
        "prompt": "vistool_with_img_info_v2",
        "exe_code": True,
        "temperature": 0.6,
        "client_type": "openai",
        "api_name": "gpt-4.1"
    }
    
    # Use a single executor for the batch (more efficient)
    executor = PythonExecutor(use_process_isolation=True)
    
    try:
        results = []
        for i, data in enumerate(batch_data):
            print(f"\nProcessing item {i+1}/{len(batch_data)}...")
            try:
                messages, response = evaluate_single_data(args, data, client, executor)
                results.append({
                    "question": data["question"],
                    "response": response,
                    "messages": messages
                })
                # Reset executor state between items if needed
                executor.reset()
            except Exception as e:
                print(f"Error processing item {i+1}: {str(e)}")
                results.append({
                    "question": data["question"],
                    "error": str(e)
                })
        
        # Save batch results
        with open("./test_data/batch_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\nBatch processing complete. Results saved to ./test_data/batch_results.json")
        
    finally:
        # Cleanup
        del executor


if __name__ == "__main__":
    # Check if running with command-line arguments
    if len(sys.argv) > 1:
        main_with_args()
    else:
        # Run simple demo
        main()