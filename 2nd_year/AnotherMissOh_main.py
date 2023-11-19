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

warnings.simplefilter(action='ignore', category=UserWarning)

logger = logging.getLogger(__name__)
CE = torch.nn.CrossEntropyLoss()
Softmax = nn.Softmax(dim=1)

def build_parser():
	parser = argparse.ArgumentParser(description='object extraction from dramaQA data annotation')

	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--bsz', type=int, default=32)
	parser.add_argument('--lr_PLM', type=float, default=1e-6)
	parser.add_argument('--lr_classifier', type=float, default=1e-3)
	parser.add_argument('--weight_decay', type=float, default=0.01)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--save_criterion', type=str, choices=['loss', 'acc'])

	parser.add_argument('--bbox_threshold', type=float, default=0.3)
	parser.add_argument('--rel_threshold', type=float, default=0.7)

	parser.add_argument('--train_distributed', action='store_true')
	parser.add_argument('--logging', action='store_true')
	parser.add_argument('--do_test', action='store_true')
	parser.add_argument('--do_train', action='store_true')

	parser.add_argument('--run_name', type=str, default='debugging')
	parser.add_argument('--project_name', type=str, default='LBA')
	parser.add_argument('--wandb', action='store_true')

	parser.add_argument('--custom_dataset_dir', type=str, default='./saves/custom_dataset')
	parser.add_argument('--output_dir', type=str, default='./saves')
	parser.add_argument('--dataset_dir', type=str, default='./dataset')

	parser.add_argument('--model_name', type=str, default='bert-base-uncased', choices=['bert-base-uncased'])
	parser.add_argument('--max_length', type=int, default=512)

	# parser.add_argument('--train_question', type=str, default='AnotherMissOh_train_created_data.pkl')

	parser.add_argument('--world_size', default=1, type=int,
						help='number of distributed processes')
	parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')	   
	parser.add_argument('--fp16', action='store_true')

	parser.add_argument('--model_ckpt', type=str)
	parser.add_argument('--classifier_head_ckpt', type=str)

	parser.add_argument('--do_preprocess', action='store_true')
	
	parser.add_argument('--preprocessed_train_data', type=str, 
					 	help='if you have done preprocess and want to use that data, you should input the data path')
	parser.add_argument('--preprocessed_valid_data', type=str, 
					 	help='if you have done preprocess and want to use that data, you should input the data path')
	parser.add_argument('--preprocessed_test_data', type=str, 
					 	help='if you have done preprocess and want to use that data, you should input the data path')

	return parser

def train(args, model, classifier_head, optimizer_PLM, optimizer_classifier, train_data, global_step):
	model.train()
	classifier_head.train()

	total_loss = 0.
	steps = torch.tensor(len(train_data)).to(args.device)

	for batch_idx, batch in enumerate(tqdm(train_data)):
		global_step += 1
		optimizer_PLM.zero_grad()
		optimizer_classifier.zero_grad()
		# batch = utils_sys.dict_to_device(batch, args.device)
		outputs = model(batch[0].to(args.device), attention_mask=batch[1].to(args.device), token_type_ids=batch[2].to(args.device))['last_hidden_state']
		target_word_idx = batch[3] 
		output_of_interest = []
		for idx, instance in enumerate(outputs):
			key = list(target_word_idx[idx].keys())[0]
			
			try:
				output_of_interest_instance = instance[target_word_idx[idx][key]]
			except:
				ipdb.set_trace()
			#average 
			output_of_interest_instance = torch.mean(output_of_interest_instance, dim=0)
			output_of_interest_instance = output_of_interest_instance.view(-1,output_of_interest_instance.shape[0])
			output_of_interest.append(output_of_interest_instance)
		#
		output_of_interest = torch.cat(output_of_interest, dim = 0)
		output_classifier = classifier_head(output_of_interest)
		labels = batch[4]
		labels = labels.to(args.device)

		loss = CE(output_classifier, labels)
		loss.backward()

		optimizer_PLM.step()
		optimizer_classifier.step()
		
		total_loss += loss.item()
	
	return total_loss/steps, global_step

def eval(args, model, classifier_head, val_data):
	model.eval()
	classifier_head.eval()

	total_loss = 0.
	total_acc = 0.
	steps = torch.tensor(len(val_data)).to(args.device)

	for batch_idx, batch in enumerate(val_data):
		outputs = model(batch[0].to(args.device), attention_mask=batch[1].to(args.device), token_type_ids=batch[2].to(args.device))['last_hidden_state']

		target_word_idx = batch[3] 
		output_of_interest = []
		for idx, instance in enumerate(outputs):
			key = list(target_word_idx[idx].keys())[0]
			output_of_interest_instance = instance[target_word_idx[idx][key]]
			#average 
			output_of_interest_instance = torch.mean(output_of_interest_instance, dim=0)
			output_of_interest_instance = output_of_interest_instance.view(-1,output_of_interest_instance.shape[0])
			output_of_interest.append(output_of_interest_instance)
		#
		output_of_interest = torch.cat(output_of_interest, dim = 0)
		output_classifier = classifier_head(output_of_interest)
		labels = batch[4]
		labels = labels.to(args.device)

		loss = CE(output_classifier, labels)

		prob = Softmax(output_classifier)
		prediction = torch.argmax(prob, dim=1)
		correct = torch.sum(labels == prediction)
		acc = correct / labels.size(0)

		total_loss += loss
		total_acc += acc

	return total_loss/steps, total_acc/steps

def main(args):
# ============================================= run name ======================================================
	if args.run_name == 'date':
		args = utils_sys.set_run_name(args)

# ============================================= gpu setting ======================================================
	args = utils_sys.set_gpu(args)

# ============================================= seed setting ======================================================
	if args.seed:
	 	utils_sys.set_seed(args, args.seed)
	
# ============================================= log setting ======================================================
	if args.do_train and args.logging:

		log_dir  = os.path.join(args.output_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)

		if utils_sys.is_main_process() or not args.distributed():
			args.log_file = os.path.join(log_dir, args.run_name+'.log')
			logging.basicConfig(filename=args.log_file,
								format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
								datefmt = '%m/%d/%Y %H:%M:%S',
								level = logging.INFO ,
								filemode='w')
			logger.info("device: {}, distributed training: {}, 16-bits training: {}".format(
				args.device, bool(args.distributed), args.fp16))
			logger.info(args)		

# ============================================= wandb setting ======================================================
	if args.do_train and args.wandb:
		if (utils_sys.is_main_process() or not dist.is_initialized()):
			wandb.init(
						# set the wandb project where this run will be logged
						project=args.project_name,
						mode = "disabled",
						name=args.run_name
					)
# ============================================= prepare module ======================================================
	if (utils_sys.is_main_process() or not dist.is_initialized()):
		print('loading model...')
		model, tokenizer = utils.get_model(args)
		model.to(args.device)

		classifier_head = utils.Predicate2Bool(model.config.hidden_size)
		classifier_head.to(args.device)

		if args.do_test:
			if not (args.model_ckpt or args.classifier_head_ckpt):
				raise AssertionError("checkpoints are not provided!")
				
			model.load_state_dict(torch.load(args.model_ckpt))
			model.to(args.device)

			classifier_head.load_state_dict(torch.load(args.classifier_head_ckpt))
			classifier_head.to(args.device)


		if args.do_train:
			#optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = beta)
			no_decay = ["bias", "LayerNorm.weight"]
			optimizer_grouped_parameters = [
				{
					"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
					"weight_decay": args.weight_decay,
				},
				{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
			]
			
			optimizer_PLM = optim.AdamW(optimizer_grouped_parameters, lr=args.lr_PLM)
			optimizer_classifier = optim.AdamW(classifier_head.parameters(), lr=args.lr_classifier)

	if (utils_sys.is_main_process() or not dist.is_initialized()):
		print('loading model finished')
# ============================================= loading dataset  ======================================================
	if (utils_sys.is_main_process() or not dist.is_initialized()):
		print('loading data...')
	
	if args.do_train:
		
		if args.do_preprocess:
			train_questions = utils_sys.read_json(os.path.join(args.dataset_dir, 'DramaQA/AnotherMissOhQA_train_set.json'))
			val_questions = utils_sys.read_json(os.path.join(args.dataset_dir, 'DramaQA/AnotherMissOhQA_val_set.json'))
			sg_fpath = os.path.join(args.dataset_dir, 'AnotherMissOh', 'scene_graph')
			
			train_answerability_data = utils_sys.read_pkl(os.path.join(args.custom_dataset_dir, 'AnotherMissOh_train_created_data.pkl'))
			val_answerability_data = utils_sys.read_pkl(os.path.join(args.custom_dataset_dir, 'AnotherMissOh_val_created_data.pkl'))
			
			train_sg2sentence = {}
			for question in tqdm(train_questions):
				train_sg = utils.AnotherMissOh_sg(args, sg_fpath, question)
				train_sg2sentence[question['qid']] = ". ".join(train_sg)
			
			# for debugging
			# train_sg2sentence_subset = utils_sys.get_subset(train_sg2sentence)
			# tokenized_train_sg2sentence = utils.sentence2tokenize(args, tokenizer, train_sg2sentence_subset)

			tokenized_train_sg2sentence = utils.sentence2tokenize(args, tokenizer, train_sg2sentence)
			new_train_questions = utils.preprocess_question(args, train_answerability_data)
			inputs = utils.input_preprocess_V2(args, tokenizer, new_train_questions, tokenized_train_sg2sentence)
			train_data = inputs

			val_sg2sentence = {}
			for question in tqdm(val_questions):
				train_sg = utils.AnotherMissOh_sg(args, sg_fpath, question)
				val_sg2sentence[question['qid']] = ". ".join(train_sg)
			
			# for debugging
			# val_sg2sentence_subset = utils_sys.get_subset(val_sg2sentence)
			# tokenized_val_sg2sentence = utils.sentence2tokenize(args, tokenizer, val_sg2sentence_subset)

			tokenized_val_sg2sentence = utils.sentence2tokenize(args, tokenizer, val_sg2sentence)
			new_val_questions = utils.preprocess_question(args, val_answerability_data)
			inputs = utils.input_preprocess_V2(args, tokenizer, new_val_questions, tokenized_val_sg2sentence)
			val_data = inputs
		else:
			if not (args.preprocessed_train_data or args.preprocessed_valid_data):
				raise AssertionError("preprocessed data are not provided!")
			train_data = utils_sys.read_pkl(args.preprocessed_train_data)
			val_data = utils_sys.read_pkl(args.preprocessed_valid_data)


# ============================================= training code ======================================================
	if args.do_train:
		best_loss = None
		best_acc = None

		global_step = 0
		args.total_step = len(train_data)*args.epochs

		for epoch in range(args.epochs):
			if (utils_sys.is_main_process() or not dist.is_initialized()):
				logger.info("start epoch {} training...".format(epoch+1))
				print("start epoch {} training...".format(epoch+1))
			training_loss, global_step = train(args, model, classifier_head, optimizer_PLM, optimizer_classifier, train_data, global_step)

			with torch.no_grad():
				if (utils_sys.is_main_process() or not dist.is_initialized()):
					logger.info("start epoch {} evaluating...".format(epoch+1))

				eval_loss, eval_acc = eval(args, model, classifier_head, val_data)

			if args.distributed:
				torch.distributed.barrier()

			if (utils_sys.is_main_process() or not dist.is_initialized()):
				print("Epoch [{}/{}], Step [{}/{}]  training loss: {:.5f},  eval loss : {:.5f},  eval acc : {:.5f}".format(
					epoch+1, args.epochs, global_step, args.total_step, training_loss, eval_loss, eval_acc))
				if args.wandb:
					wandb.log({"Epoch": epoch+1, "train_loss": training_loss, "validation_loss": eval_loss, "validation_acc": eval_acc})
				if args.logging:
					logger.info("Epoch [{}/{}], Step [{}/{}]  training loss: {:.5f},  eval loss : {:.5f},  eval acc : {:.5f}".format(
					epoch+1, args.epochs, global_step, args.total_step, training_loss, eval_loss, eval_acc))
				
				
				if args.save_criterion == 'loss':
					if best_loss == None or eval_loss <= best_loss:
						best_loss = eval_loss
						best_acc = eval_acc
						best_epoch = epoch+1
						print("model save for epoch {}".format(epoch+1))
						logger.info("model save for epoch {}".format(epoch+1))

						ckpt_dir  = os.path.join(args.output_dir, 'ckpt')
						os.makedirs(ckpt_dir, exist_ok=True)

						model_fname = os.path.join(ckpt_dir, args.run_name+'_best_loss.pt')
						classifier_fname = os.path.join(ckpt_dir, args.run_name+'_classifier_best_loss.pt')

						with open(model_fname, 'wb') as f:
							torch.save(model.state_dict(), f)
							sys.stdout.flush()
						with open(classifier_fname, 'wb') as f:
							torch.save(classifier_head.state_dict(), f)
							sys.stdout.flush()
					else:
						print("model not saved for epoch {}".format(epoch+1))
						print("best eval acc : {}	  best eval loss : {}	  current eval loss : {}	(@ epoch: {})".format(best_acc, best_loss, eval_loss, best_epoch))

				elif args.save_criterion == 'acc':
					if best_acc == None or eval_acc >= best_acc:
						best_loss = eval_loss
						best_acc = eval_acc
						best_epoch = epoch+1
						print("model save for epoch {}".format(epoch+1))
						logger.info("model save for epoch {}".format(epoch+1))

						ckpt_dir  = os.path.join(args.output_dir, 'ckpt')
						os.makedirs(ckpt_dir, exist_ok=True)

						model_fname = os.path.join(ckpt_dir, args.run_name+'_best_acc.pt')
						classifier_fname = os.path.join(ckpt_dir, args.run_name+'_classifier__best_acc.pt')

						with open(model_fname, 'wb') as f:
							torch.save(model.state_dict(), f)
							sys.stdout.flush()
						with open(classifier_fname, 'wb') as f:
							torch.save(classifier_head.state_dict(), f)
							sys.stdout.flush()
					else:
						print("model not saved for epoch {}".format(epoch+1))
						print("best eval acc : {}	  best eval loss : {}	  current eval loss : {}	(@ epoch: {})".format(best_acc, best_loss, eval_loss, best_epoch))

			if args.distributed:
				torch.distributed.barrier()	

# ============================================= test code ======================================================
	if args.do_test:
		
		if args.do_preprocess:
			test_questions = utils_sys.read_json(os.path.join(args.dataset_dir, 'DramaQA/AnotherMissOhQA_test_set.json'))
			sg_fpath = os.path.join(args.dataset_dir, 'AnotherMissOh', 'scene_graph')

			test_answerability_data = utils_sys.read_pkl(os.path.join(args.custom_dataset_dir, 'AnotherMissOh_test_created_data.pkl'))

			test_sg2sentence = {}
			for question in tqdm(test_questions):
				test_sg = utils.AnotherMissOh_sg(args, sg_fpath, question)
				test_sg2sentence[question['qid']] = ". ".join(test_sg)

			tokenized_test_sg2sentence = utils.sentence2tokenize(args, tokenizer, test_sg2sentence)
			new_test_questions = utils.preprocess_question(args, test_answerability_data)
			inputs = utils.input_preprocess_V2(args, tokenizer, new_test_questions, tokenized_test_sg2sentence)
			test_data = inputs
		else:
			if not args.preprocessed_test_data:
				raise AssertionError("preprocessed data are not provided!")
			test_data = utils_sys.read_pkl(args.preprocessed_test_data)

		with torch.no_grad():
			eval_loss, eval_acc = eval(args, model, classifier_head, test_data)
		print("test loss : {}	  test acc : {}".format(eval_loss, eval_acc))
	return

if __name__ == '__main__':
	
	if not torch.cuda.is_available():
		print("Need available GPU(s) to run this model...") #
		quit() #

	parser = build_parser()
	args = parser.parse_args()

	main(args)

	ipdb.set_trace()