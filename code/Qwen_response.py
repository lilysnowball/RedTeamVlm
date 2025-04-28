from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from utils import create_dataset
import csv


# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# use cuda device

dataset_path = "./data/DM_instructions.csv"
dataset = create_dataset(dataset_path)

def generate_response(model_path = "/root/autodl-tmp/model/Qwen-VL-Chat",dataset = dataset):
  model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  outputs = []
  # Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
  # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
  for sample in dataset:
    output = {}
    query = tokenizer.from_list_format([
        {'image': sample['img']},
        {'text': sample['txt']},
    ])
    '''
    # below code is for Qwen-VL (the pretrained model)
    inputs = tokenizer(query, return_tensors='pt', add_special_tokens=True)
    # Print information about inputs
    
    print(f"Type of inputs: {type(inputs)}")
    print(f"Shape of inputs: {inputs.shape if hasattr(inputs, 'shape') else 'No shape attribute'}")
    print(f"Content of inputs: {inputs}")
    

    # inputs = inputs.to(model.device)
    input_ids = inputs.input_ids[:, :-1].to(model.device)
    attention_mask = inputs.attention_mask[:, :-1].to(model.device)
    
    with torch.inference_mode():
      pred = model.generate(
      input_ids = input_ids,
      attention_mask = attention_mask,
      max_new_tokens=1000,      
      eos_token_id=tokenizer.eos_token_id,
  )
    prompt_len = input_ids.shape[-1]
    generated_ids = pred[0, prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(response)
    '''
    response, history = model.chat(tokenizer, query=query, history=None)
    # print(response)
    output['response'] = response
    output['scenario'] = sample['scenario']
    output['index'] = sample['index']
    outputs.append(output)
  return outputs




if __name__ == "__main__":
    model_path = "/root/autodl-tmp/model/Qwen-VL-Chat"
    dataset_path = "./data/DM_instructions.csv" # only csv is supported for Qwen-VL now
    dataset = []
    outputs = []
    dataset = create_dataset(dataset_path)
    dataset = dataset[:300]
    outputs = generate_response(model_path,dataset)
    # save the response to a csv file
    with open('./response/Qwen-VL_response_DM_instructions.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['index','scenario','response'])
        for item in outputs:            
            writer.writerow([item['index'], item['scenario'], item['response']])

    print("Successfully saved the response to ./response/Qwen-VL_response_DM_instructions.csv")





