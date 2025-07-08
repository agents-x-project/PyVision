import json
import sys
import os
import argparse
from openai import OpenAI
from inference_engine.vis_inference_demo_gpt import evaluate_single_data, evaluate_single_with_cleanup
from inference_engine.safe_persis_shared_vis_python_exe import PythonExecutor

def main():
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
    
    # messages, final_response = evaluate_single_with_cleanup(eval_args, data, client)
    executor = PythonExecutor()
    messages, final_response = evaluate_single_data(eval_args, data, client, executor)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.save_messages:
        messages_path = os.path.join(args.output_dir, "test_messages.json")
        with open(messages_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
        print(f"Messages saved to: {messages_path}")
    
    # Print response
    print("\n" + "="*50)
    print("Final Response:")
    print("="*50)
    print(final_response)
    print("="*50 + "\n")


if __name__ == "__main__":
    main()