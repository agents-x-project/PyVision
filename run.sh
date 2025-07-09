# openai client

python main.py \
    --image_path ./test_data/one_image_demo.png \
    --question "What is the color of the liquid contained in the glass on the table?" \
    --api_config ./api_config_files/api_config_openai.json \
    --client_type openai \
    --prompt_template ./prompt_template/prompt_template_vis.json \
    --prompt vistool_with_img_info_v2 \
    --exe_code \
    --max_tokens 10000 \
    --temperature 0.6 \
    --output_dir ./test_data \
    --save_messages 

# azure client

# python main.py \
#     --image_path ./test_data/one_image_demo.png \
#     --question What is the color of the liquid contained in the glass on the table? \
#     --api_config ./api_config_files/api_config_azure.json \
#     --client_type azure \
#     --prompt_template ./prompt_template/prompt_template_vis.json \
#     --prompt vistool_with_img_info_v2 \
#     --exe_code \
#     --max_tokens 10000 \
#     --temperature 0.6 \
#     --output_dir ./test_data \
#     --save_messages 

# vllm client 

# python main.py \
#     --image_path ./test_data/one_image_demo.png \
#     --question What is the color of the liquid contained in the glass on the table? \
#     --api_config ./api_config_files/api_config_vllm.json \
#     --client_type vllm \
#     --prompt_template ./prompt_template/prompt_template_vis.json \
#     --prompt vistool_with_img_info_v2 \
#     --exe_code \
#     --max_tokens 10000 \
#     --temperature 0.6 \
#     --output_dir ./test_data \
#     --save_messages 