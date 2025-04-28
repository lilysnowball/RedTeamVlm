from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
import os
import csv
from utils import create_dataset

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
    
def generate_response(model_path,dataset):
    model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
    processor = InstructBlipProcessor.from_pretrained(model_path,use_fast=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for sample in dataset:      
        output = {}
        if sample['img'] is not None:
            image = load_image(sample['img'])
            inputs = processor(images=image, text=sample['txt'], return_tensors="pt").to(device)
        else:
            inputs = processor(text=sample['txt'], return_tensors="pt").to(device)
        response = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=500,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        generated_text = processor.batch_decode(response, skip_special_tokens=True)[0].strip()
        output['response'] = generated_text
        # print(output['response'])
        output['scenario'] = sample['scenario']
        output['index'] = sample['index']
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    dataset_path = "./data/SIUO/siuo_new.csv"
    model_path = "/root/autodl-tmp/model/instructblip-vicuna-7b"
    mode = "puretext"
    dataset = []
    outputs = []
    dataset = create_dataset(dataset_path,mode)
    outputs = generate_response(dataset,model_path)
    # save the response to a csv file
    save_path = f'./result/{mode}/instructblip_generate.csv'
    # Make directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['index','scenario','response'])
        for item in outputs:            
            writer.writerow([item['index'], item['scenario'], item['response']])

            
            
    