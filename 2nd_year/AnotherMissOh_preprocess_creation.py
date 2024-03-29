import json
import argparse
import torch
import ipdb
import utils
import os
import pickle
import itertools
import sys
import copy
import numpy as np
import random
import logging
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Tuple, Union, Dict, List, Optional
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

import utils_sys
import utils

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import warnings
import wandb 

def build_parser():
	parser = argparse.ArgumentParser(description='data preprocseeing for train and test (optional)')

	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--bsz', type=int, default=32)

	parser.add_argument('--model_name', type=str, default='bert-base-uncased', choices=['bert-base-uncased'])
	parser.add_argument('--max_length', type=int, default=512)

	parser.add_argument('--bbox_threshold', type=float, default=0.3)
	parser.add_argument('--rel_threshold', type=float, default=0.7)

	parser.add_argument('--custom_dataset_dir', type=str, default='./saves/custom_dataset')
	parser.add_argument('--output_dir', type=str, default='./saves')
	parser.add_argument('--dataset_dir', type=str, default='../dataset')
	parser.add_argument('--data_split', type=str, required=True, choices=['train', 'val', 'test'])

	parser.add_argument('--save_name', type=str, required=True, 
					 	help='file name or detailed path for the processed file (e.g., preprocessed_data/train.pkl)')
	return parser

def main(args):
	if not args.output_dir:
		raise AssertionError("directory is not provided!")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir, exist_ok=True)

	_, tokenizer = utils.get_model(args)

	questions = utils_sys.read_json(os.path.join(args.dataset_dir, f'DramaQA/AnotherMissOhQA_{args.data_split}_set.json'))
	sg_fpath = os.path.join(args.dataset_dir, 'AnotherMissOh', 'scene_graph')

	answerability_data = utils_sys.read_pkl(os.path.join(args.custom_dataset_dir, f'AnotherMissOh_{args.data_split}_created_data.pkl'))

	sg2sentence = {}
	for question in tqdm(questions):
		dataset_sg = utils.AnotherMissOh_sg(args, sg_fpath, question)
		sg2sentence[question['qid']] = ". ".join(dataset_sg)


	tokenized_sg2sentence = utils.sentence2tokenize(args, tokenizer, sg2sentence)
	new_train_questions = utils.preprocess_question(args, answerability_data)
	inputs = utils.input_preprocess_V3(args, tokenizer, new_train_questions, tokenized_sg2sentence)

	save_fname = os.path.join(args.output_dir, args.save_name)
	with open(save_fname, 'wb') as f:
		pickle.dump(inputs, f)

	return


if __name__ == '__main__':
	
	if not torch.cuda.is_available():
		print("Need available GPU(s) to run this model...") #
		quit() #

	parser = build_parser()
	args = parser.parse_args()

	main(args)