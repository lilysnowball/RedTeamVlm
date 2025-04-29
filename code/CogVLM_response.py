import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import os
import base64
from io import BytesIO
import csv  
from utils import *

def load_image(image_input):
    """
    Load image from various input formats:
    - URL string (starting with 'http://' or 'https://')
    - Local file path
    - Bytes or BytesIO object
    - Base64 encoded string
    - String representation of bytes (b'...')
    
    Args:
        image_input: Image input in any of the above formats
        
    Returns:
        PIL.Image: Loaded RGB image, or None if loading fails
    """
    try:
        # Case 1: URL
        if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
            try:
                response = requests.get(image_input, stream=True)
                response.raise_for_status()  # Raise an error for bad status codes
                return Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                print(f"Error loading image from URL: {e}")
                return None

        # Case 2: String representation of bytes
        if isinstance(image_input, str) and image_input.startswith('b\''):
            try:
                image_input = eval(image_input)  # Convert string representation to bytes
            except Exception as e:
                print(f"Error evaluating bytes string: {e}")
                return None

        # Case 3: Local file path
        if isinstance(image_input, str) and os.path.isfile(image_input):
            try:
                return Image.open(image_input).convert('RGB')
            except Exception as e:
                print(f"Error loading local file: {e}")
                return None

        # Case 4: Base64 encoded string
        if isinstance(image_input, str):
            try:
                image_data = base64.b64decode(image_input)
                return Image.open(BytesIO(image_data)).convert('RGB')
            except Exception as e:
                print(f"Error decoding base64: {e}")
                return None

        # Case 5: Bytes or BytesIO
        if isinstance(image_input, (bytes, BytesIO)):
            try:
                if isinstance(image_input, bytes):
                    image_input = BytesIO(image_input)
                return Image.open(image_input).convert('RGB')
            except Exception as e:
                print(f"Error loading from bytes: {e}")
                return None

        print("Unsupported image input format")
        return None

    except Exception as e:
        print(f"Unexpected error loading image: {e}")
        return None 


def generate_response(dataset,model_path):
    tokenizer = LlamaTokenizer.from_pretrained('/root/autodl-tmp/model/vicuna-7b-v1.5') # only download the tokenizer file
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to('cuda').eval()
    outputs = []
    for sample in dataset:
        output = {}
        if sample['img'] is not None:
            image = load_image(sample['img'])
            inputs = model.build_conversation_input_ids(tokenizer, query=sample['txt'], history=[], images=[image])  # chat mode
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
                'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
            }
            gen_kwargs = {"max_length": 2048, "do_sample": False}
            with torch.no_grad():
                response = model.generate(**inputs, **gen_kwargs)
                response = response[:, inputs['input_ids'].shape[1]:]
        else:
            # pure-text
            txt_inputs = tokenizer(sample['txt'], return_tensors="pt",return_token_type_ids=True).to("cuda")
            '''
            txt_inputs = tokenizer(sample['txt]), return_tensors="pt").to("cuda")
            if "token_type_ids" not in txt_inputs:
            txt_inputs["token_type_ids"] = torch.zeros_like(txt_inputs["input_ids"])
            '''
            with torch.no_grad():
                # generate only 512 new tokens beyond the prompt
                response = model.generate(
                **txt_inputs,
                max_new_tokens=512,
                do_sample=False
                )
            response = response[:, txt_inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(response[0], skip_special_tokens=True) 
        output['response'] = response
        # print(output['response'])
        output['scenario'] = sample['scenario']
        output['index'] = sample['index']
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    # dataset_path = "./data/SIUO/siuo_new.json"
    dataset_path = "./data/few_shot.json"
    model_path = "/root/autodl-tmp/model/cogvlm-chat-hf"
    for mode in ["redundantimg","irrelevantimg"]:
        dataset = []
        outputs = []
        dataset = create_safe_dataset(dataset_path,mode)
        outputs = generate_response(dataset,model_path)
        # save the response to a csv file
        save_path = f'./few_shot_result/safe/{mode}/cogvlm_generate.csv'
        # Make directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['index','scenario','response'])
            for item in outputs:            
                writer.writerow([item['index'], item['scenario'], item['response']])