import torch
import torch.nn.functional as N
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import re
import json
import time

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
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_images_from_bytes(image_data_list):
    return [load_image_from_bytes(data) for data in image_data_list]

def prepare_imgs_tensor_both_cases(sample):       
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

def measure_refusal_strength(model_path,datasets):
    refusal_lst = [
        "Sorry", "sorry",
        "unfortunately", "unfortunate", "sadly",
        "explicit", "deadly", "crim", "criminal", "illegal", "dangerous", "harmful", "warning", "alarm", "caution",
        "shame", "conspiracy",
        "Subject", "contrary", "shouldn"
    ]
    vocab_size = 32000
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

    refusal_token_ids = []
    for token in refusal_lst:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]                
        refusal_token_ids.append(token_id)    
    # refusal vector
    token_one_hot = torch.zeros(vocab_size)
    for token_id in refusal_token_ids:
        token_one_hot[token_id] = 1.0

    lm_head = model.lm_head    
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        norm = model.transformer.ln_f
    else:
        raise ValueError(f"Incorrect Model") 

    F_all = []
    for sample in datasets:
        if sample['img'] == None:
            prompt = construct_conv_prompt(sample)
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )   
            with torch.no_grad():   
                outputs = model(input_ids, images=None, image_sizes=None, output_hidden_states=True)        
        else:
            prompt = construct_conv_prompt(sample)
            input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )      
            images_tensor, images_size = prepare_imgs_tensor_both_cases(sample)
            with torch.no_grad(): 
                outputs = model(input_ids, images=images_tensor, image_sizes=images_size, output_hidden_states=True)  
        F = []       
        for i, r in enumerate(outputs.hidden_states[1:]):                   
            layer_output = norm(r)                    
            logits = lm_head(layer_output)                 
            next_token_logits = logits[:, -1, :]
            reference_logits = token_one_hot.to(next_token_logits.device)                 
            cos_sim = N.cosine_similarity(next_token_logits, reference_logits)
            F.append(cos_sim.item())    
        F_all.append(F)
    return F_all

if __name__ == "__main__":
    model_path = "/root/autodl-tmp/model/llava-v1.6-7b-"
    datasets = create_safe_dataset(dataset_path,mode)
    F_all = measure_refusal_strength(model_path,datasets)
    print(F_all)
                  
       
  