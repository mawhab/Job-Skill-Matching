import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from common import defaults
from common.preprocessing_utils import ValidationData, IdToName
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_args():
    parser = argparse.ArgumentParser(description="Generate job descriptions using a pretrained model.")
    parser.add_argument('--distribution', type=str, default='train', help='Choose the distribution of jobs to generate descriptions for: train, val', choices=['train', 'val'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for description generation')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.distribution == 'train':
        jobs = IdToName.get_all_job_names()
    elif args.distribution == 'val':
        jobs = ValidationData.get_queries_texts()
    
    batch_size = args.batch_size

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16), device_map='auto')


    generated_descriptions = {}
    num_batches = int(np.ceil(len(jobs) / batch_size))

    print(f"Starting description generation for {len(jobs)} titles in {num_batches} batches of size {batch_size}...")

    with torch.inference_mode():
        for i in tqdm(range(num_batches//2, num_batches), desc="Generating Descriptions"):
            batch_titles = jobs[i * batch_size : (i + 1) * batch_size]

            if not batch_titles:
                continue

            batch_messages = []
            for title in batch_titles:
                user_prompt_content = f"Job Title: {title}"
                messages = [
                    {"role": "system", "content": defaults.LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt_content},
                ]
                batch_messages.append(messages)

            templated_prompts = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in batch_messages
            ]

            inputs = tokenizer(
                templated_prompts,
                return_tensors="pt",
                padding=True, 
                truncation=False
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id 
            )

            for idx, title in enumerate(batch_titles):
                input_length = inputs.input_ids.shape[1] 
                response_ids = outputs[idx, input_length:]

                description = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                generated_descriptions[title] = description

            torch.cuda.empty_cache()
        
    with open(f'data/{args.distribution}_descriptions.json', 'w') as f:
        json.dump(generated_descriptions, f)