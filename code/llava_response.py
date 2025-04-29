import argparse
from time import sleep
import torch
import torch.nn.functional as F
import pandas as pd
import csv
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import re
import gc
from torch.cuda.amp import autocast
import base64
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from utils import *

def generate_response(model_path,dataset):
    dataset: list[dict]
    sample: dict
    outputs: list[dict]

    outputs = []

    model_name = get_model_name_from_path(model_path)   
    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16    
    }

    tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    **kwargs
    )       
    
    def find_conv_mode(model_name):
        # select conversation mode based on the model name
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"  
        return conv_mode    
        
    def adjust_query_for_images(qs):   
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        return qs

    def construct_conv_prompt(sample):        
        conv = conv_templates[find_conv_mode(model_name)].copy()  
        if (sample['img'] != None):     
            qs = adjust_query_for_images(sample['txt'])
        else:
            qs = sample['txt']
        conv.append_message(conv.roles[0], qs)  
        conv.append_message(conv.roles[1], None)       
        prompt = conv.get_prompt()       
        return prompt

    def load_image(image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(image_files):
        out = []
        for image_file in image_files:
            image = load_image(image_file)
            out.append(image)
        return out
    
    def load_image_from_bytes(image_data):   
        if isinstance(image_data, str) and image_data.startswith('b\''):
            # Remove the b'' wrapper if it's stored as string representation of bytes
            image_data = eval(image_data)       
        try:
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            try:
                image_data = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_data)).convert("RGB")
                return image
            except Exception as e2:
                print(f"Failed to process image.")
                return None
            
    def load_images_from_bytes(image_data_list):       
        return [load_image_from_bytes(data) for data in image_data_list]

    def prepare_imgs_tensor(sample):
        try:
            # Case 1: Comma-separated file paths
            if isinstance(sample['img'], str):
                image_files_path = sample['img'].split(",")
                img_prompt = load_images(image_files_path)
            # Case 2: Single binary image
            elif isinstance(sample['img'], bytes):
                img_prompt = [load_image_from_bytes(sample['img'])]
            # Case 3: List of binary images
            elif isinstance(sample['img'], list):
                # Check if all elements in the list are bytes
                if all(isinstance(item, bytes) for item in sample['img']):
                    img_prompt = load_images_from_bytes(sample['img'])
                else:
                    raise ValueError("List contains non-bytes data.")
            else:
                raise ValueError("Unsupported data type in sample['img']. "
                                "Expected str, bytes, or list of bytes.")
            # Compute sizes
            images_size = [img.size for img in img_prompt if img is not None]
            # Process images into tensor
            images_tensor = process_images(img_prompt, image_processor, model.config)
            images_tensor = images_tensor.to(model.device, dtype=torch.float16)
            return images_tensor, images_size
        except Exception as e:
            print(f"Error preparing image tensors: {e}")
            return None, None
 
    for sample in dataset:      
        output = {}
        prompt = construct_conv_prompt(sample)           
        if sample['img'] != None:
            images_tensor, images_size = prepare_imgs_tensor(sample)
        else:
            images_tensor = None
            images_size = None           
        input_ids_txt = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()  
        # print(f"input_ids_txt shape: {input_ids_txt.shape}")      
        with torch.inference_mode():
            output_ids_txt = model.generate(
                input_ids_txt,
                images= images_tensor,
                image_sizes= images_size,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=1000,
                use_cache=True,
            )        
        # Decode the output
        output['response'] = tokenizer.batch_decode(output_ids_txt, skip_special_tokens=True)[0].strip() 
        output['scenario'] = sample['scenario']
        output['index'] = sample['index']
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    # dataset_path = "./data/SIUO/siuo_new.json"
    dataset_path = "./data/few_shot.json"
    model_path = "/root/autodl-tmp/model/llava-v1.6-vicuna-7b"
    for mode in ["puretext","figimg","typoimg","vcd","redundantimg","irrelevantimg"]:
        dataset = []
        outputs = []
        dataset = create_unsafe_dataset(dataset_path,mode)
        outputs = generate_response(model_path,dataset)
        # save the response to a csv file
        save_path = f'./few_shot_result/unsafe/{mode}/llava_generate.csv'
        # Make directory if it does not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['index','scenario','response'])
            for item in outputs:            
                writer.writerow([item['index'], item['scenario'], item['response']])
        print(f"Results saved to {save_path}")

                    
                    
