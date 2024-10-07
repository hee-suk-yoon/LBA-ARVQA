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
    required=True,
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

    file = open(args.pred_path)
    pred_contents = json.load(file)
    output_save = []

    output_dir = os.path.join(args.home_path, "result")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_fname = os.path.join(output_dir, args.save_name + '.json')

    for idx, pred_item in enumerate(tqdm(pred_contents)):
        question = pred_item["question"]
        answer = pred_item["gt_answer"]
        pred = pred_item["pred"]

        response_message, response_dict = annotate(args, question, pred)
        import ipdb; ipdb.set_trace()
        save_item = {
            "vid": pred_item["vid"],
            "question": question,
            "gt_answer": answer,
            "pred": pred,
            "answerable": response_dict["answerable"],
            "reasoning": response_dict["reasoning"],

        }
        output_save.append(save_item)


    # Write combined content to a json file
    with open(save_fname, "w") as f:
        json.dump(output_save, f, indent=4)
    print("Process completed!")


if __name__ == "__main__":
    main()

