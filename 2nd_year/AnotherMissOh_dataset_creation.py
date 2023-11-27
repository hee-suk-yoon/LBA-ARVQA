import json
import argparse
import torch
import ipdb
import utils
import os
import pickle
import itertools
import copy
import numpy as np
import random
from tqdm import tqdm

import utils_sys
import utils

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def build_parser():
	parser = argparse.ArgumentParser(description='object extraction from dramaQA data annotation')

	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--bsz', type=int, default=32)
	
	parser.add_argument('--run_name', type=str, default='debugging')

	parser.add_argument('--data_name', type=str, choices=['train', 'val', 'test'])

	parser.add_argument('--output_dir', type=str, default='./saves/custom_dataset')
	parser.add_argument('--dataset_dir', type=str, default='../dataset')

	parser.add_argument('--world_size', default=1, type=int,
						help='number of distributed processes')
	parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')	   

	return parser

def main(args):
# ============================================= run name ======================================================
	if args.run_name == 'date':
		args = utils_sys.set_run_name(args)

# ============================================= gpu setting ======================================================
	args = utils_sys.set_gpu(args)

# ============================================= seed setting ======================================================
	if args.seed:
	 	utils_sys.set_seed(args, args.seed)

# ============================================= load question and object list ======================================
	if not args.output_dir:
		raise AssertionError("directory is not provided!")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir, exist_ok=True)
	
	# questions = utils_sys.read_json(os.path.join(args.dataset_dir, 'DramaQA/AnotherMissOhQA_train_set.json'))
	questions = utils_sys.read_json(os.path.join(args.dataset_dir, 'DramaQA/AnotherMissOhQA_' + args.data_name + '_set.json'))
	
	object_list = utils_sys.read_pkl(os.path.join(args.output_dir, 'AnotherMissOhQA_object_list.pkl'))

	exception_object = ['Haeyoung', 'Jinsang', 'Taejin', 'relationship', 'kind', 'something', 'communication', 'someone', 'everything', 'com', 'color']

	new_data = {}
	current_idx = 0

	
	for idx, question_dict in enumerate(tqdm(questions)):
		current_idx += 0
		question = question_dict['que'] 

		extracted_objects_ = []

		word_tokenized_question = word_tokenize(question)
		try:
			extracted_objects = utils.object_extract(question)
		except:
			continue

		for object_, [object_idx1, object_idx2] in extracted_objects:
		# for object_item in extracted_objects:
			exception_list = [1 if exception in object_ else 0 for exception in exception_object]
			if not sum(exception_list) > 0:
				extracted_objects_.append([object_, [object_idx1, object_idx2]])

		if len(extracted_objects_) == 0:
			continue

		n = len(extracted_objects_)
		lsts = list(itertools.product([0, 1], repeat=n))

		if question_dict['qid'] not in list(new_data.keys()):
			new_data[question_dict['qid']] = []
		
		for lst in lsts: #go through each augmentation case
			#making the 'red' list (refer to reserach notebook for what 'red' list is)
			new_word_tokenized_question = copy.deepcopy(word_tokenized_question)
			new_wordtokenize_idx = copy.deepcopy(extracted_objects_)

			for idx, word_bool in enumerate(lst): 
				#if word_bool is true (that is no augmentation)
				if word_bool:
					continue
				else: #if word_bool is false(we need to augment)
					wordtokenize_idx_single = new_wordtokenize_idx[idx]
					
					object_ = [wordtokenize_idx_single[0]]
					object_wordtokenize_idx = wordtokenize_idx_single[1]
					object_wordtokenize_idx_start = object_wordtokenize_idx[0]
					object_wordtokenize_idx_end = object_wordtokenize_idx[1]

					while True: #TODO for the TODO is the bottom of filtering 
						new_sampled_object_ = random.choice(object_list)
						
						#TODO later we need to develop how to make sure that the new sampled object is not same as the original.
						new_word_tokenized_question[object_wordtokenize_idx_start:object_wordtokenize_idx_end] = new_sampled_object_

						#update new wordtokenize_idx
						reference_idx = object_wordtokenize_idx_start
						length_diff = len(object_) - len(new_sampled_object_) 
						for idx2, wordtokenize_idx_single_update in enumerate(new_wordtokenize_idx):
							if idx == idx2:
								new_wordtokenize_idx[idx2][0] = new_sampled_object_
								new_wordtokenize_idx[idx2][1][1] -= length_diff

							else: #need to update the idx for other objects 
								if new_wordtokenize_idx[idx2][1][0] > reference_idx: #TODO This is assuming no overlap, which I think is reasonable
									new_wordtokenize_idx[idx2][1][0] -= length_diff #update start idx
									new_wordtokenize_idx[idx2][1][1] -= length_diff #update end idx
						break
			
			new_data_instance = [new_word_tokenized_question, new_wordtokenize_idx,list(lst)]
			new_data[question_dict['qid']].append(new_data_instance)
	
	save_fname = 'AnotherMissOh_' + args.data_name + '_created_data.pkl'
	utils_sys.save_pkl(new_data, os.path.join(args.output_dir, save_fname))

	return







if __name__ == "__main__":

	if not torch.cuda.is_available(): #
		print("Need available GPU(s) to run this model...") #
		quit() #

	parser = build_parser()
	args = parser.parse_args()

	main(args)

	# ipdb.set_trace()
