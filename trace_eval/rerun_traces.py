import openai
import json
import os
import re
import argparse
import asyncio
import concurrent.futures
from tqdm import tqdm
import dotenv
import sys

# Make toolkit-local imports work when this script is executed directly.
TOOLKIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOOLKIT_DIR not in sys.path:
    sys.path.insert(0, TOOLKIT_DIR)

from utils.qa_em import extract_solution, compute_score_em
import torch
import transformers

dotenv.load_dotenv()

class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False

def process_batch_local(batch_lines, model, tokenizer, think_after_information=False):
    """
    Processes a batch of lines from the input JSONL file using a local model.
    """
    prompts = []
    original_data_list = []
    prompt_texts = []
    results = [None] * len(batch_lines)

    for i, line_content in enumerate(batch_lines):
        try:
            data = json.loads(line_content)
            original_sequence = data['sequences_str']

            question = data['question']
            
            stop_tag = "</information>"
            prompt_end_index = original_sequence.rfind(stop_tag)
            prompt_text = original_sequence[:prompt_end_index] + "</information>"
            
            # answer_start_tag = '<answer>'
            # split_lines = original_sequence.splitlines()
            # prompt_end_index = split_lines[-1].rfind(answer_start_tag)
            
            # if prompt_end_index == -1:
            #     results[i] = line_content.strip()
            #     continue
            
            # split_lines[-1] = split_lines[-1][:prompt_end_index]
            # prompt_text = "\n".join(split_lines)

            if think_after_information:
                user_prompt_content = f"<|im_start|>user \n Please think step by step after you have read the information inside <information> and </information>. Give your thinking process in <think> and </think>. Then give your answer to {question} in <answer> and </answer>.<|im_end|>"
                full_prompt = prompt_text + user_prompt_content + "\n<|im_start|>assistant\n"
            else: 
                user_prompt_content = f"<|im_start|>user \n Please directly give your answer to \"{question}\" in <answer> and </answer>. e.g. <answer> Beijing </answer> no other words.<|im_end|>"
                full_prompt = prompt_text + user_prompt_content + "\n<|im_start|>assistant\n"
            
            print(f"debug {i} full_prompt: ", full_prompt)
            prompts.append(full_prompt)
            original_data_list.append(data)
            prompt_texts.append(prompt_text)

        except json.JSONDecodeError:
            print(f"Skipping line due to JSON decode error: {line_content.strip()}")
            results[i] = None
        except Exception as e:
            print(f"An unexpected error occurred processing line: {line_content[:100]}... Error: {e}")
            results[i] = line_content.strip()

    if not prompts:
        return [r for r in results if r]

    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, padding=True, return_tensors='pt').to(model.device)
        max_attempts = 10
        
        stop_sequences = ["</answer>", " </answer>"]
        stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(stop_sequences, tokenizer)])
        
        result_idx = 0
        for i in range(len(batch_lines)):
            if results[i] is not None:
                continue
                
            # Get single prompt inputs
            single_inputs = {k: v[result_idx:result_idx+1] for k,v in inputs.items()}
            
            generated_text = ""
            attempts = 0
            
            while not generated_text and attempts < max_attempts:
                outputs = model.generate(
                    **single_inputs,
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.2,
                    stopping_criteria=stopping_criteria,
                )

                generated_tokens = outputs[:, single_inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                attempts += 1
            
            print("================================================")
            print(f"debug {i} generated_text: ", generated_text)
            print(f"attempts: {attempts}")
            print("================================================")
            
            prompt_text = prompt_texts[result_idx]
            data = original_data_list[result_idx]
            
            new_sequence = prompt_text + generated_text + "</answer>"
            reward = compute_score_em(new_sequence, data["ground_truth"])

            is_contaminated = False
            original_sequence = data['sequences_str']
            assistant_start_marker = '<|im_start|>assistant'
            assistant_start_index = original_sequence.rfind(assistant_start_marker)
            if assistant_start_index != -1:
                agent_actions = original_sequence[assistant_start_index:]
                has_search = '<search>' in agent_actions
                is_correct = False
                ground_truths = data["ground_truth"] 
                for gt in ground_truths["target"]:
                    if gt in original_sequence:
                        is_correct = True
                        break
                if not has_search and is_correct:
                    is_contaminated = True
       
            new_data = {
                "question": data["question"],
                "sequences_str": new_sequence,
                "ground_truth": data["ground_truth"],
                "reward": reward,
                "answer": extract_solution(new_sequence),
                "is_contaminated": is_contaminated,
                "do_search": data.get("do_search", False),
            }
            
            results[i] = json.dumps(new_data)
            result_idx += 1

    except Exception as e:
        print(f"An unexpected error occurred during batch processing: {e}")
        # Fill in remaining unprocessed lines with original content
        result_idx = 0
        for i in range(len(batch_lines)):
            if results[i] is None:
                results[i] = batch_lines[i].strip()


    return [r for r in results if r]


def process_line(line_content, client, model, think_after_information=False):
    """
    Processes a single line from the input JSONL file.
    This function is intended to be run in a thread pool.
    It reruns the evaluation for a single sample.
    """
    try:
        data = json.loads(line_content)
        original_sequence = data['sequences_str']
        
        is_contaminated = False
        assistant_start_marker = '<|im_start|>assistant'
        assistant_start_index = original_sequence.rfind(assistant_start_marker)
        if assistant_start_index != -1:
            agent_actions = original_sequence[assistant_start_index:]
            has_search = '<search>' in agent_actions
            is_correct = False
            ground_truths = data["ground_truth"] 
            for gt in ground_truths["target"]:
                if gt in original_sequence:
                    is_correct = True
                    break
            if not has_search and is_correct:
                is_contaminated = True

        top_tag = "</information>"
        prompt_end_index = original_sequence.rfind(top_tag)
        prompt_text = original_sequence[:prompt_end_index] + top_tag
        
        # answer_start_tag = '<answer>'
        # # Use rfind to get the last occurrence, which is the actual answer
        # prompt_end_index = original_sequence.rfind(answer_start_tag)
        
        # if prompt_end_index == -1:
        #     # If there's no answer tag, we can't process it. Return original content.
        #     return line_content.strip()

        # prompt_text = original_sequence[:prompt_end_index]

        messages = []
        system_match = re.search(r"<\|im_start\|>system\n(.*?)\n<\|im_end\|>", prompt_text, re.DOTALL)
        user_match = re.search(r"<\|im_start\|>user\n(.*?)\n<\|im_end\|>", prompt_text, re.DOTALL)
        assistant_match = re.search(r"<\|im_start\|>assistant\n(.*)", prompt_text, re.DOTALL)

        if system_match:
            messages.append({"role": "system", "content": system_match.group(1).strip()})
        if user_match:
            messages.append({"role": "user", "content": user_match.group(1).strip()})
        
        if assistant_match:
            assistant_prompt_content = assistant_match.group(1).strip()
            messages.append({"role": "assistant", "content": assistant_prompt_content})

        if think_after_information:
            user_prompt_content = f"ONLY USE THE INFORMATION INSIDE <information> AND </information> TO ANSWER THE QUESTION. Please think step by step after you have read the information inside <information> and </information>. Give your thinking process in <think> and </think>. Then give your answer to \"{data['question']}\" in <answer> and </answer>. If you don't have enough information from the given information, you can say <answer> I don't know </answer>, but give your reason what the information is missing in <reason> and </reason>."
            messages.append({"role": "user", "content": user_prompt_content})
        else: 
            user_prompt_content = f"ONLY USE THE INFORMATION INSIDE <information> AND </information> TO ANSWER THE QUESTION. Please directly give your **short** answer to \"{data['question']}\" in <answer> and </answer>. e.g. <answer> Beijing </answer> no other words. If you don't have enough information from the given information, you can say <answer> I don't know </answer>, but give your reason what the information is missing in <reason> and </reason>."
            messages.append({"role": "user", "content": user_prompt_content})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0
                # stop=["</answer>"],
            )
            generated_text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": generated_text})
            
            new_sequence = prompt_text + generated_text + "</answer>"
            reward = compute_score_em(new_sequence, data["ground_truth"])
       
            new_data = {
                "question": data["question"],
                "sequences_str": new_sequence,
                "ground_truth": data["ground_truth"],
                "messages": messages,
                "reward": reward,
                "answer": extract_solution(new_sequence),
                "is_contaminated": is_contaminated,
                "do_search": data.get("do_search", False),
            }
            
            return json.dumps(new_data)

        except Exception as e:
            print(f"An error occurred calling OpenAI API for line: {line_content[:100]}... Error: {e}")
            return line_content.strip()

    except json.JSONDecodeError:
        print(f"Skipping line due to JSON decode error: {line_content.strip()}")
        return None # This will be filtered out.
    except Exception as e:
        print(f"An unexpected error occurred in process_line for line: {line_content[:100]}... Error: {e}")
        return line_content.strip()

async def main_async(args):
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    if args.local_inference:
        if not args.model_name:
            print("Error: --model_name must be specified for local inference.")
            return
        print("Loading local model for inference...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded.")
    else:
        try:
            client = openai.OpenAI(base_url=os.getenv("OPENAI_BASE"), api_key=os.getenv("OPENAI_API_KEY"))
        except openai.OpenAIError:
            print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return

    with open(args.input_file, 'r') as infile:
        lines = infile.readlines()
        if args.limit:
            lines = lines[:args.limit]
    
    if not lines:
        print("No lines to process.")
        return

    print(f"Starting evaluation of {len(lines)} items from {args.input_file}")
    if args.output_file:
        print(f"Results will be saved to: {args.output_file}")

    processed_count = 0
    reward_sum = 0
    
    if args.local_inference:
        with open(args.output_file, 'w') as output_file_handle:
            for i in tqdm(range(0, len(lines), args.local_batch_size), desc="Rerunning evaluation (local, batched)"):
                batch_lines = lines[i:i+args.local_batch_size]
                results = process_batch_local(batch_lines, model, tokenizer, args.think_after_information)
                for result in results:
                    output_file_handle.write(result + '\n')
                    try:
                        reward = json.loads(result)["reward"]
                        reward_sum += reward
                        processed_count += 1
                    except (json.JSONDecodeError, KeyError):
                        # This can happen if the line was an error and returned as is.
                        pass
    else:
        loop = asyncio.get_running_loop()
        
        if args.batch_size <= 0:
            max_workers = os.cpu_count() or 1
        else:
            max_workers = args.batch_size
        
        with open(args.output_file, 'w') as output_file_handle:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        loop.run_in_executor(executor, process_line, line, client, args.model_name, args.think_after_information)
                        for line in lines
                    ]

                    for future in tqdm(asyncio.as_completed(futures), total=len(futures), desc="Rerunning evaluation"):
                        result = await future
                        if result:
                            output_file_handle.write(result + '\n')
                            try:
                                reward = json.loads(result)["reward"]
                                reward_sum += reward
                                processed_count += 1
                            except (json.JSONDecodeError, KeyError):
                                pass
            except Exception as e:
                print(f"An error occurred during concurrent processing: {e}")

    print("\n--- Rerun Summary ---")
    print(f"Total items attempted: {len(lines)}")
    print(f"Total items successfully written to output: {processed_count}")
    print(f"Output saved to: {args.output_file}")
    if processed_count > 0:
        print(f"Average reward (test score): {reward_sum / processed_count}")
    else:
        print("Average reward (test score): N/A (no valid processed items)")

def main():
    parser = argparse.ArgumentParser(description="Rerun evaluation on a JSONL file with parallel processing.")
    parser.add_argument("input_file", help="The input JSONL file.")
    parser.add_argument("output_file", help="The output JSONL file.")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of parallel API calls. A safe default is 10.")
    parser.add_argument("--local_batch_size", type=int, default=32, help="Batch size for local inference.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of items to process from the input file.")
    parser.add_argument("--think_after_information", action="store_true", help="Prompt the model to think before answering.")
    parser.add_argument("--local_inference", action="store_true", help="Use local inference instead of API calls.")
    parser.add_argument("--model_name", type=str, default=None, help="Model ID for local inference.")
    args = parser.parse_args()
    
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main() 
