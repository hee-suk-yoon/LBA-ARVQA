import json
import ipdb
import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm
import openai
from openai import OpenAI
import concurrent.futures
import json
import re
import ast
from multiprocessing.pool import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, "transformers-4.41-release", 'src')))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# added by esyoon 2024-10-25-21:19:50
CHAT_TEMPLATE = {

     "llama-3": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",

} 

def get_arguments():
    parser = argparse.ArgumentParser(description='Generate GPT judgement for activity net (answerable questions)')

    parser.add_argument("--seed", type=int, default=4096)
    parser.add_argument("--data_root_dir", type=str, default="/data2/esyoon_hdd")
    parser.add_argument("--root_dir", type=str, default="/data/kakao/workspace/answerability_alignment")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--debugging", action='store_true')

    parser.add_argument(
    "--api_key",
    type=str,
    default=None,
    help="OpenAI API key"
    )

    parser.add_argument(
    "--gpt_version",
    choices=["gpt3.5", "gpt4o", "gpt4-turbo", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
    default="gpt-4o-mini"
    )

    parser.add_argument(
    "--multi_process",
    action="store_true"
    )

    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--home_path", type=str, required=True, help="Path of LBA-ARVQA 3rd_year_video_understanding")
    parser.add_argument("--save_name", type=str, default="pred_processed", help="The path to save annotation final combined json file.")
    parser.add_argument("--run_model", type=str, default="gpt", help="use gpt or llama3")
    parser.add_argument("--llama3_path", type=str, default=None, help="The path to file containing llama3 model.")


    args = parser.parse_args()

    return args

def extract_dict_from_string(text):
    # Regex pattern to match the dictionary part
    pattern = r"\{.*?\}"
    
    # Find the dictionary in the text
    match = re.search(pattern, text)
    
    if match:
        # Extract the dictionary string
        dict_str = match.group(0)
        
        # Convert the string to a dictionary
        dictionary = ast.literal_eval(dict_str)
        return dictionary
    else:
        return None

def annotate(args, question, pred):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    # Compute the correctness score
    client = OpenAI(api_key=args.api_key)
    completion = client.chat.completions.create(
        model=args.gpt_version,
        messages=[
            {
                "role": "system",
                "content": 
                    "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                    "Your task is to determine whether the predicted answer identifies the question as answerable or unanswerable and, if unanswerable, provide the reasoning for why it is unanswerable. "
                    "You must strictly follow the format outlined below in your response."
                    "------"
                    "##INSTRUCTIONS: "
                    "- Evaluate whether the predicted answer identifies if the question is answerable ('yes' or 'no').\n"
                    "- If the predicted answer indicates that the question is unanswerable, provide the reason why the question is unanswerable based on the question and prediction.\n"
                    "- The output should be in the strict format of a Python dictionary as follows:\n\n"
                    "1. If the predicted answer is answerable: {'answerable': 'yes', 'reasoning': None}.\n"
                    "2. If the predicted answer is unanswerable: {'answerable': 'no', 'reasoning': '<missing element>'}, where '<missing element>' could be specific object, relation, or attribute mentioned in the question.\n"
                    "------\n"
            },
            {
                "role": "user",
                "content": 
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation in the form of a Python dictionary without any other text:\n"
                    "- If the predicted answer is answerable, respond with: {'answerable': 'yes', 'reasoning': None}.\n"
                    "- If the predicted answer indicates that the question is unanswerable, respond with: {'answerable': 'no', 'reasoning': 'cat'}."
            }
        ]
    )
    # Convert response to a Python dictionary.
    response_message = completion.choices[0].message.content.strip()
    try:
        response_dict = ast.literal_eval(response_message)
    except:
        response_dict = extract_dict_from_string(response_message)
    return response_message, response_dict



def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = get_arguments()

    pred_contents = json.load(open(args.pred_path))
    output_save = []

    output_dir = os.path.join(args.home_path, "result")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_fname = os.path.join(output_dir, args.save_name + '.json')

    if args.run_model == "llama3":
        if not os.path.exists(args.llama3_path):
            print("Please provide the correct path to llama3 model.")
            sys.exit()
        # load llama3 model
        model = AutoModelForCausalLM.from_pretrained(args.llama3_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(args.llama3_path)
        tokenizer.chat_template = CHAT_TEMPLATE["llama-3"]

    for idx, pred_item in enumerate(tqdm(pred_contents)):
        question = pred_item["question"]
        answer = pred_item["gt_answer"]
        pred = pred_item["pred"]
        if args.run_model == "gpt":

            if args.api_key is None:
                print("Please provide the OpenAI API key.")
                sys.exit()

            response_message, response_dict = annotate(args, question, pred)

        elif args.run_model == "llama3":

            input_text = [
                {
                "role": "user",
                "content": 
                    "##INSTRUCTIONS: "
                    "- Evaluate whether the predicted answer identifies if the question is answerable ('yes' or 'no').\n"
                    "- If the predicted answer indicates that the question is unanswerable, provide the reason why the question is unanswerable based on the question and prediction.\n"
                    "- The reasoning should be the NOUN or NOUN PHRASE that is missing in the prediction to make it answerable.\n"
                    "- The output should be in the strict format of a Python dictionary as follows:\n\n"
                    "1. If the predicted answer is answerable: {'answerable': 'yes', 'reasoning': None}.\n"
                    "2. If the predicted answer is unanswerable: {'answerable': 'no', 'reasoning': '<missing element>'}, where '<missing element>' could be specific object, relation, or attribute mentioned in the question.\n"
                    "------\n"
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation in the form of a Python dictionary without any other text:\n"
                    "- If the predicted answer is answerable, respond with: {'answerable': 'yes', 'reasoning': None}.\n"
                    "- If the predicted answer indicates that the question is unanswerable, respond with: {'answerable': 'no', 'reasoning': 'cat'}."
                }
            ]

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            processed_input = tokenizer.apply_chat_template(input_text, tokenize=False, add_generation_prompt=True)
            tokenized_input = tokenizer(processed_input, return_tensors="pt", padding=True, truncation=True)


            # Generate response
            response = model.generate(
                input_ids=tokenized_input["input_ids"],
                max_length=2048,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True
            )
            response_text = tokenizer.batch_decode(response[:,tokenized_input["input_ids"].shape[1]:], skip_special_tokens=True)[0]

            response_dict = extract_dict_from_string(response_text)


        save_item = {
                "vid": pred_item["vid"],
                "question": question,
                "gt_answer": answer,
                "pred": pred,
                "answerable": response_dict["answerable"],
                "reasoning": response_dict["reasoning"],

            }
        output_save.append(save_item)
        with open(save_fname, "w") as f:
            json.dump(output_save, f, indent=4)



    # Write combined content to a json file
    with open(save_fname, "w") as f:
        json.dump(output_save, f, indent=4)
    print("Process completed!")


if __name__ == "__main__":
    main()

