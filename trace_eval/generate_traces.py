import argparse
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import re
from tqdm import tqdm
import sys
import os

# Make toolkit-local imports work when this script is executed directly.
TOOLKIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOOLKIT_DIR not in sys.path:
    sys.path.insert(0, TOOLKIT_DIR)

from utils.qa_em import compute_score_em
import numpy as np
import pickle
from utils.model_loading import select_checkpoint_path, _normalize_local_path

FEW_SHOT_TEMPLATE = """<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and 
it will return the top searched results between <information> and </information>. You can search as many times as your want. 
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, 
without detailed illustrations. For example, <answer> Beijing </answer>.
This is a few-shot learning exercise. Examples are provided below.
Question: What is the birth date of the lead singer of Coldplay?<think>I need to find out the lead singer of Coldplay and their birth date.</think><search>lead singer of Coldplay birth date</search><information>Doc 1(Title: Chris Martin) Christopher Anthony John Martin (born 2 March 1977) is an English singer, songwriter, and musician. He is the lead singer, pianist, rhythm guitarist, and co-founder of the rock band Coldplay.
</information><think>The lead singer of Coldplay is Chris Martin, and he was born on March 2, 1977.</think><answer>March 2, 1977</answer>

Question: What is the most populous city in the United States?
<think>I need to determine which city in the United States has the largest population.</think><search>most populous city in the United States</search>
<information>Doc 1(Title: New York City) New York City is the most populous city in the United States, with an estimated population of over 8.3 million people.
</information><think>New York City is the most populous city in the United States.</think><answer>New York City</answer>

Question: {question}
<|im_end|>
"""


ZERO_SHOT_TEMPLATE = """<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}<|im_end|>
"""

# Chain of Thought (CoT) - Step-by-step reasoning without search
COT_TEMPLATE = """<|im_start|>user
Answer the following question by thinking step by step. Show your reasoning process clearly, then provide the final answer.

Question: {question}

Please think step by step with <think> and </think> tags.
If you find you lack some knowledge, you can call a search engine by <search> query </search> and 
it will return the top searched results between <information> and </information>. You can search as many times as your want. 
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, 
without detailed illustrations. For example, <answer> Beijing </answer>.
This is a few-shot learning exercise. Examples are provided below.
Question: What is the birth date of the lead singer of Coldplay?<think>I will think step by step. First, I need to find out the lead singer of Coldplay. Then, I need to find out their birth date.</think><search>lead singer of Coldplay</search>
<information>Doc 1(Title: Chris Martin) Christopher Anthony John Martin is an English singer, songwriter, and musician. He is the lead singer, pianist, rhythm guitarist, and co-founder of the rock band Coldplay.
</information>
<think>I need to find out their birth date. The lead singer of Coldplay is Chris Martin</think><search>Chris Martin birth date</search>
<information>Doc 1(Title: Chris Martin) Christopher Anthony John Martin (born 2 March 1977) is an English singer, songwriter, and musician. He is the lead singer, pianist, rhythm guitarist, and co-founder of the rock band Coldplay.
</information<answer>March 2, 1977</answer>

Question: {question}
<|im_end|>
"""

# ReAct (Reflecting + Acting) - Explicit reasoning-action loop with search support
REACT_TEMPLATE = """<|im_start|>user
Answer the given question using the Reflecting + Acting approach: Reflect step by step, then take actions (like searching) when needed, observe the results, reflect on what you've learned, and continue reasoning until you can provide the final answer.

You can use <search> query </search> to search for information, which will return results between <information> and </information>. 
When you have enough information, provide your final answer using <answer> your answer </answer>.

The following example shows how to use Reflecting + Acting.

Question: What is the birth date of the lead singer of Coldplay?
<think>I want to find the birth date of Coldplay's lead singer. First, I should find out who the lead singer is.</think>
<search>lead singer of Coldplay</search>
<information>Doc 1(Title: Chris Martin) Christopher Anthony John Martin is an English singer, songwriter, and musician. He is the lead singer, pianist, rhythm guitarist, and co-founder of the rock band Coldplay.</information>
<think>Reflect: I have found that the lead singer of Coldplay is Chris Martin. Now I need to find Chris Martin's birth date.</think>
<search>Chris Martin birth date</search>
<information>Doc 1(Title: Chris Martin) Christopher Anthony John Martin (born 2 March 1977) is an English singer, songwriter, and musician. He is the lead singer, pianist, rhythm guitarist, and co-founder of the rock band Coldplay.</information>
<think>Reflect: I have found Chris Martin's birth date is March 2, 1977. Now I can provide the answer.</think>
<answer>March 2, 1977</answer>

Question: {question}
<|im_end|>
"""


# Structured Reasoning - More formal reasoning structure with explicit example
STRUCTURED_REASONING_TEMPLATE = """<|im_start|>user
Answer the following question using structured reasoning. Break down the problem into steps, analyze each step, and synthesize the final answer.
If you find you lack some knowledge, you can call a search engine by <search> query </search> and 
it will return the top searched results between <information> and </information>. You can search as many times as your want. 

The following example shows how to use structured reasoning:

Question: What is the birth date of the lead singer of Coldplay?

Reasoning Steps:
1. Understanding the question: The question asks about the birth date of the lead singer of Coldplay.
2. Identifying what information is needed: I need to know who the lead singer of Coldplay is, and then find their birth date.
3. Applying knowledge or searching for information:
<think>I need to find the lead singer of Coldplay.</think>
<search>lead singer of Coldplay</search>
<information>Doc 1(Title: Chris Martin) Christopher Anthony John Martin is an English singer, songwriter, and musician. He is the lead singer, pianist, rhythm guitarist, and co-founder of the rock band Coldplay.</information>
<think>The lead singer of Coldplay is Chris Martin. Now I need to find his birth date.</think>
<search>Chris Martin birth date</search>
<information>Doc 1(Title: Chris Martin) Christopher Anthony John Martin (born 2 March 1977) is an English singer, songwriter, and musician. He is the lead singer, pianist, rhythm guitarist, and co-founder of the rock band Coldplay.</information>
<think>Chris Martin was born on 2 March 1977.</think>
<answer>March 2, 1977</answer>

---

Question: {question}

Reasoning Steps:
1. Understanding the question:
2. Identifying what information is needed:
3. Applying knowledge or searching for information:
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>

<|im_end|>
"""


PROMPT_TEMPLATES = {
    "fewshot": FEW_SHOT_TEMPLATE,
    "zeroshot": ZERO_SHOT_TEMPLATE,
    "cot": COT_TEMPLATE,
    "react": REACT_TEMPLATE,
    "structured": STRUCTURED_REASONING_TEMPLATE,
}

def _convert_numpy_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_native(elem) for elem in obj]
    return obj

def batch_search(queries: list, retriever_url, top_k):
    if not queries:
        return []
    try:
        payload = {"queries": queries, "topk": top_k, "return_scores": True}
        response = requests.post(retriever_url, json=payload)
        response.raise_for_status()
        results = response.json().get('result', [])

        def _passages2string(retrieval_result):
            format_reference = ''
            for idx, doc_item in enumerate(retrieval_result or []):
                content = doc_item.get('document', {}).get('contents', '')
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
            return format_reference
        
        if len(results) != len(queries):
            print(f"Warning: Number of search results ({len(results)}) does not match number of queries ({len(queries)}).")
            return [""] * len(queries)

        return [_passages2string(res) for res in results]
    except Exception as e:
        print(f"Error in batch search: {e}")
        return [""] * len(queries)


def execute_multi_turn(
    current_prompts: list,
    active_mask: list,
    do_search_flags: list,
    model,
    tokenizer,
    max_turns: int,
    max_new_tokens: int,
    do_search: bool,
    retriever_url: str,
    top_k: int
):
    """
    Execute multi-turn dialogue with search capability.
    
    Args:
        current_prompts: List of current prompt strings for each example
        active_mask: List of booleans indicating which examples are still active
        do_search_flags: List of booleans tracking if each example performed search
        model: The language model for generation
        tokenizer: The tokenizer
        max_turns: Maximum number of turns to execute
        max_new_tokens: Maximum tokens to generate per turn
        do_search: Whether to perform search when requested
        retriever_url: URL for the search retriever service
        top_k: Number of top results to retrieve
    
    Returns:
        tuple: (updated_current_prompts, updated_active_mask, updated_do_search_flags)
    """
    for turn in range(max_turns):
        if not any(active_mask):
            break

        active_prompts = [p for p, active in zip(current_prompts, active_mask) if active]
        active_indices = [idx for idx, active in enumerate(active_mask) if active]

        inputs = tokenizer(active_prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
        )
        
        generated_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        search_queries = []
        search_indices = []

        for i, gen_text in enumerate(generated_outputs):
            original_index = active_indices[i]
            
            processed_text = gen_text.split('</search>')[0] + '</search>' if '</search>' in gen_text else \
                             gen_text.split('</answer>')[0] + '</answer>' if '</answer>' in gen_text else \
                             gen_text
            
            current_prompts[original_index] += processed_text
            
            tag_match = re.search(r'<(search|answer)>(.*)', processed_text, re.DOTALL)
            
            if tag_match:
                action = tag_match.group(1)
                content = tag_match.group(2).strip()
                end_tag = f'</{action}>'
                if end_tag in content:
                    content = content.split(end_tag, 1)[0].strip()

                if action == 'answer':
                    active_mask[original_index] = False
                
                elif action == 'search' and do_search:
                    do_search_flags[original_index] = True
                    if content.strip():
                        search_queries.append(content)
                        search_indices.append(original_index)
                    else:
                        current_prompts[original_index] += "<information></information>\n\n"
                else:
                    active_mask[original_index] = False
            else:
                active_mask[original_index] = False
        
        if search_queries:
            search_results = batch_search(search_queries, retriever_url, top_k)
            for res_idx, original_idx in enumerate(search_indices):
                current_prompts[original_idx] += f"<information>{search_results[res_idx]}</information>\n\n"
    
    return current_prompts, active_mask, do_search_flags


def execute_multi_turn_summary(
    current_prompts: list,
    active_mask: list,
    do_search_flags: list,
    model,
    tokenizer,
    max_turns: int,
    max_new_tokens: int,
    do_search: bool,
    retriever_url: str,
    top_k: int
):
    """
    Execute multi-turn dialogue with search capability and summary support.
    
    Differences from execute_multi_turn:
    1. Supports additional action types: think, think_summary, information_summary
    2. Adds turn information and summary prompts to observations
    3. Uses more sophisticated tag parsing that handles multiple tag types
    
    Args:
        current_prompts: List of current prompt strings for each example
        active_mask: List of booleans indicating which examples are still active
        do_search_flags: List of booleans tracking if each example performed search
        model: The language model for generation
        tokenizer: The tokenizer
        max_turns: Maximum number of turns to execute
        max_new_tokens: Maximum tokens to generate per turn
        do_search: Whether to perform search when requested
        retriever_url: URL for the search retriever service
        top_k: Number of top results to retrieve
    
    Returns:
        tuple: (updated_current_prompts, updated_active_mask, updated_do_search_flags)
    """
    # Tag pattern that matches multiple action types
    tag_pattern = re.compile(
        r'<\s*(search|answer|think|think_summary|information_summary)\b[^>]*>(.*?)</\s*\1\s*>',
        re.IGNORECASE | re.DOTALL
    )
    
    for turn in range(max_turns):
        if not any(active_mask):
            break

        active_prompts = [p for p, active in zip(current_prompts, active_mask) if active]
        active_indices = [idx for idx, active in enumerate(active_mask) if active]

        inputs = tokenizer(active_prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
        )
        
        generated_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        search_queries = []
        search_indices = []

        for i, gen_text in enumerate(generated_outputs):
            original_index = active_indices[i]
            
            # Process text to stop at closing tags
            processed_text = gen_text.split('</search>')[0] + '</search>' if '</search>' in gen_text else \
                             gen_text.split('</answer>')[0] + '</answer>' if '</answer>' in gen_text else \
                             gen_text
            
            current_prompts[original_index] += processed_text
            
            # Parse action using the comprehensive tag pattern
            action = None
            content = ''
            found = False
            
            # Priority: search > answer > think_summary > information_summary > think
            for match in tag_pattern.finditer(processed_text):
                tag = match.group(1).lower()
                content_candidate = match.group(2).strip()
                
                if tag == 'search':
                    # Search requires valid content (more than 5 chars)
                    if content_candidate and len(content_candidate) > 5:
                        action = 'search'
                        content = content_candidate
                        found = True
                        break  # Use first valid search
                    else:
                        continue
                elif tag == 'answer':
                    action = 'answer'
                    content = content_candidate
                    found = True
                    break  # Use first answer
                elif tag in ['think_summary', 'information_summary']:
                    # Summary types: only accept if no search/answer found yet
                    if not found:
                        action = tag
                        content = content_candidate
                        found = True
                elif tag == 'think':
                    # Think: only accept if nothing else found
                    if not found:
                        action = 'think'
                        content = content_candidate
            
            # Handle actions
            if action == 'answer':
                active_mask[original_index] = False
            elif action == 'search' and do_search:
                do_search_flags[original_index] = True
                if content.strip():
                    search_queries.append(content)
                    search_indices.append(original_index)
                else:
                    # Empty search query - add empty information block with turn info
                    turn_info = f"\n[Turn {turn + 1}/{max_turns}] "
                    current_prompts[original_index] += f"{turn_info}user\n<information></information>\n\nassistant\n"
            elif action in ['think', 'think_summary', 'information_summary']:
                # These actions don't stop the conversation, just continue
                # Add turn information if this is near the last turn
                if max_turns and turn >= max_turns - 2:
                    turn_info = f"\n[Turn {turn + 1}/{max_turns}] "
                    sharp_message = f"\n\nThis is my LAST turn (Turn {turn + 1}/{max_turns}). I MUST provide final answer now with <answer> and </answer>.\n\n"
                    current_prompts[original_index] += f"{turn_info}{sharp_message}"
                else:
                    turn_info = f"\n[Turn {turn + 1}/{max_turns}] "
                    current_prompts[original_index] += f"{turn_info}"
            else:
                # Invalid or no action found - stop this trajectory
                active_mask[original_index] = False
        
        # Execute batch search if there are queries
        if search_queries:
            search_results = batch_search(search_queries, retriever_url, top_k)
            for res_idx, original_idx in enumerate(search_indices):
                turn_info = f"\n[Turn {turn + 1}/{max_turns}] "
                search_result = search_results[res_idx].strip()
                
                # Check if this is near the last turn
                if max_turns and turn >= max_turns - 2:
                    sharp_message = f"\n\nThis is my LAST turn (Turn {turn + 1}/{max_turns}). I MUST provide final answer now with <answer> and </answer>."
                    current_prompts[original_idx] += (
                        f"{turn_info}user\n<information>{search_result}</information>\n\nassistant\n{sharp_message}\n"
                    )
                else:
                    # Add summary prompt to encourage summarization
                    summary_prompt = (
                        "I will provide concise, high-level summaries of both my previous reasoning and the gathered information.\n"
                        "Use the <think_summary>...</think_summary> tag for my thought process summary,\n"
                        "and the <information_summary>...</information_summary> tag for key retrieved facts or evidence.\n"
                        "Focus on clarity and brevity to help guide my next response. Or I will use <answer> and </answer> to provide the final answer if information is enough."
                    )
                    current_prompts[original_idx] += (
                        f"{turn_info}user\n<information>{search_result}</information>\n\nassistant\n{summary_prompt}\n"
                    )
    
    return current_prompts, active_mask, do_search_flags

def main(args):
    # Resolve model path: either a direct HF dir/id or a checkpoint root + step
    model_path = args.model_path
    if getattr(args, "ckpt_root", None):
        model_path = select_checkpoint_path(args.ckpt_root, args.ckpt_step)
    elif model_path:
        model_path = _normalize_local_path(model_path)
    print("debugging", "model_path", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    
    df = pd.read_parquet(args.data_path)
    
    # Clean data: remove dirty questions if requested
    if args.clean_data:
        # Try to infer dataset name from data_path
        # Common patterns: data/{dataset}/process_test.parquet or similar
        dataset = None
        data_path_str = str(args.data_path)
        for ds in ["hotpotqa", "nq_search", "2wikimultihopqa", "musique", "triviaqa", "popqa", "bamboogle"]:
            if ds in data_path_str:
                dataset = ds
                break
        
        if dataset:
            dirty_questions_path = f"data/dirty_questions_{dataset}.dump"
            if os.path.exists(dirty_questions_path):
                try:
                    with open(dirty_questions_path, "rb") as f:
                        dirty_questions = pickle.load(f)
                    original_len = len(df)
                    df = df[~df["question"].isin(dirty_questions)]
                    removed_count = original_len - len(df)
                    print(f"Removed {removed_count} dirty questions from {dataset} (out of {original_len} total)")
                except Exception as e:
                    print(f"Warning: Could not load dirty questions from {dirty_questions_path}: {e}")
            else:
                print(f"Warning: Dirty questions file not found: {dirty_questions_path}")
        else:
            print(f"Warning: Could not infer dataset name from data_path '{args.data_path}'. Skipping data cleaning.")
    
    if args.num_samples:
        df = df.head(args.num_samples)
    
    questions = df['question'].tolist()
    ground_truths = df['golden_answers'].tolist()

    with open(args.output_path, 'w') as f:
        for i in tqdm(range(0, len(questions), args.val_batch_size), desc="Evaluating batches"):
            batch_questions = questions[i:i+args.val_batch_size]
            batch_ground_truths = ground_truths[i:i+args.val_batch_size]
            
            # Select prompt template
            template_name = args.prompt_template
            if template_name not in PROMPT_TEMPLATES:
                raise ValueError(f"Unknown prompt template: {template_name}. Available templates: {list(PROMPT_TEMPLATES.keys())}")
            selected_template = PROMPT_TEMPLATES[template_name]
            current_prompts = [selected_template.format(question=q) for q in batch_questions]

            # Add assistant start tag
            for i in range(len(current_prompts)):
                current_prompts[i] += '\n<|im_start|>assistant\n'

            do_search_flags = [False] * len(batch_questions)
            active_mask = [True] * len(batch_questions)

            # Choose multi-turn execution function based on args.multi_turn_type
            if args.multi_turn_type == "summary":
                current_prompts, active_mask, do_search_flags = execute_multi_turn_summary(
                    current_prompts=current_prompts,
                    active_mask=active_mask,
                    do_search_flags=do_search_flags,
                    model=model,
                    tokenizer=tokenizer,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    do_search=args.do_search,
                    retriever_url=args.retriever_url,
                    top_k=args.top_k
                )
            else:  # default to "standard"
                current_prompts, active_mask, do_search_flags = execute_multi_turn(
                    current_prompts=current_prompts,
                    active_mask=active_mask,
                    do_search_flags=do_search_flags,
                    model=model,
                    tokenizer=tokenizer,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    do_search=args.do_search,
                    retriever_url=args.retriever_url,
                    top_k=args.top_k
                )
            
            for idx in range(len(batch_questions)):
                current_prompts[idx] += '<|im_end|>' # Add end tag
                gt_dict = {'target': batch_ground_truths[idx]}
                score = compute_score_em(solution_str=current_prompts[idx], ground_truth=gt_dict)
                serializable_ground_truth = _convert_numpy_to_native(gt_dict)
                
                result_item = {
                    "question": batch_questions[idx],
                    "sequences_str": current_prompts[idx],
                    "ground_truth": serializable_ground_truth,
                    "reward": score,
                    "do_search": do_search_flags[idx]
                }
                f.write(json.dumps(result_item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=False, default=None,
                        help="HF repo ID or local model dir. Ignored if --ckpt_root is set.")
    parser.add_argument("--ckpt_root", type=str, default=None,
                        help="Training run root containing global_step_* subdirs, or a direct model dir.")
    parser.add_argument("--ckpt_step", type=str, default="latest",
                        help="Checkpoint step to load (integer) or 'latest'.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--retriever_url", type=str, default="http://localhost:8000/retrieve")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--val_batch_size", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--do_search", action='store_true')
    parser.add_argument("--fewshot", action='store_true', help="Enable few-shot prompting. (Deprecated: use --prompt_template instead)")
    parser.add_argument("--prompt_template", type=str, default="zeroshot", 
                        choices=["fewshot", "zeroshot", "cot", "react", "structured"],
                        help="Prompt template to use. Available: fewshot, zeroshot, cot, react, structured")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate on for debugging.")
    parser.add_argument("--multi_turn_type", type=str, default="standard", choices=["standard", "summary"],
                        help="Type of multi-turn execution: 'standard' (basic search/answer) or 'summary' (with think/summary tags support).")
    parser.add_argument("--clean_data", action='store_true',
                        help="Remove dirty questions from the dataset. Requires data/dirty_questions_{dataset}.dump file to exist.")
    args = parser.parse_args()
    
    # Backward compatibility: if --fewshot is set, override prompt_template
    if args.fewshot:
        args.prompt_template = "fewshot"
    
    main(args) 
