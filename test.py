import json
import argparse
# from this import d
import torch
import ipdb
import utils
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from torch import nn, optim
import math
#for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

def inference_demo(model, classifier_head, inputs):  
	
	outputs = model(inputs[0].to(device), attention_mask=inputs[1].to(device), token_type_ids=inputs[2].to(device))['last_hidden_state']

	predicate_bool_list = []
	for idx, output in enumerate(outputs):
		predicate_bool = {}
		for idx2, predicate in enumerate(inputs[3][idx]):
			#avg_output = output[]
			avg_output = torch.mean(output[inputs[3][idx][predicate][0]:inputs[3][idx][predicate][-1]+1],dim=0).view(1,-1)
			classifier_output = classifier_head(avg_output)
			predicate_bool[predicate] = bool(torch.argmax(classifier_output).item())

		predicate_bool_list.append(predicate_bool)

	return predicate_bool_list

def main(args):

	model, tokenizer = utils.get_model(args.model_name)
	#model.load_state_dict(torch.load('/mnt/hdd/hsyoon/workspace/LBA-ARVQA/checkpoints/bert9.pt'))
	model.load_state_dict(torch.load('checkpoints/bert9.pt'))
	model.to(device)

	classifier_head = utils.Predicate2Bool(model.config.hidden_size)
	#classifier_head.load_state_dict(torch.load('/mnt/hdd/hsyoon/workspace/LBA-ARVQA/checkpoints/classifier9.pt'))
	classifier_head.load_state_dict(torch.load('checkpoints/classifier9.pt'))
	classifier_head.to(device)

	train_scenegraphs = utils.load_json(args.train_scenegraph)
	val_scenegraphs = utils.load_json(args.val_scenegraph)

	#train_questions = utils.load_pickle(args.train_question)
	#val_questions = utils.load_pickle(args.val_question)


	# train_scenegraph_ids = list(train_scenegraphs.keys())
	# train_sg2sentence = {train_scenegraph_id : utils.sg2sentence(train_scenegraphs[train_scenegraph_id]) for train_scenegraph_id in train_scenegraph_ids} #convert frame scene graph into a natural language sentence
	# train_sg2sentence_subset = get_subset(train_sg2sentence)
	# tokenized_train_sg2sentence = utils.sentence2tokenize(args, tokenizer, train_sg2sentence_subset)
	# new_train_questions = utils.preprocess_question(args, train_questions)
	# inputs = utils.input_preprocess_V2(args, tokenizer, new_train_questions, tokenized_train_sg2sentence)
	# ipdb.set_trace()
	#with open('/mnt/hdd/hsyoon/workspace/LBA-ARVQA/gqa_data/preprocessed_data_val_fixed.pkl', 'rb') as f:
	with open('gqa_data/preprocessed_data_val_fixed.pkl', 'rb') as f:
		inputs = pickle.load(f)
	test_data = inputs

	#train/dev/test split
	total_batch_size = len(inputs)
	train_batch_size = int(total_batch_size*0.8)
	val_batch_size = int((total_batch_size - train_batch_size)/2)
	test_batch_size = total_batch_size-train_batch_size-val_batch_size

	#train_data = inputs[:train_batch_size]
	#val_data = inputs[train_batch_size:train_batch_size+val_batch_size]
	#test_data = inputs[train_batch_size+val_batch_size:]

	#for getting the image id
	#with open('/mnt/hdd/hsyoon/workspace/OOD/VQRR/imageid_index.pkl', 'rb') as f:
	#	inputs = pickle.load(f)

	#id_train_data = inputs[:train_batch_size]
	#id_val_data = inputs[train_batch_size:train_batch_size+val_batch_size]
	#id_test_data = inputs[train_batch_size+val_batch_size:]

	total_data_size = 0
	total_correct = 0

	answerability_check_data = {}

	# for batch in test_data:
	# 	for idx, image_id in enumerate(batch[-1]):
	# 		if image_id in answerabilty_check_data.keys():
	# 			question = tokenizer.decode(batch[0][idx]).split('[SEP]')[1]



	with torch.no_grad(): 
		model.eval()
		classifier_head.eval()		
		for forward_step_val, batch_val in enumerate(test_data):
			#ipdb.set_trace()
			outputs = model(batch_val[0].to(device), attention_mask=batch_val[1].to(device), token_type_ids=batch_val[2].to(device))['last_hidden_state']
			#get target word features
			target_word_idx = batch_val[3] 
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
			confidence, prediction = torch.max(output_classifier, dim = 1)

			labels = batch_val[4]
			labels = labels.to(device)
			train_acc = torch.sum(prediction == labels)

			total_data_size += labels.size()[0]
			total_correct += train_acc.item()

			# ipdb.set_trace()
			
			ipdb.set_trace()
			for idx, image_id in enumerate(batch_val[-1]):
				question_idx = torch.nonzero(torch.mul(batch_val[0][idx], batch_val[2][idx]), as_tuple=True)
				question = batch_val[0][idx][question_idx].tolist()

				obj_idxes = list(batch_val[3][idx].values())[0]
				obj_idx1 = obj_idxes[0]
				obj_idx2 = obj_idxes[-1]

				if image_id in answerability_check_data.keys():
					if question in answerability_check_data[image_id][0]:
						question_idx_ = answerability_check_data[image_id][0].index(question)
						answerability_check_data[image_id][1][question_idx_].append([obj_idx1,obj_idx2])
						answerability_check_data[image_id][2][question_idx_].append(labels[idx].to(cpu).tolist())
						answerability_check_data[image_id][3][question_idx_].append(prediction[idx].to(cpu).tolist())

					else:
						answerability_check_data[image_id][0].append(question)
						answerability_check_data[image_id][1].append([[obj_idx1, obj_idx2]])
						answerability_check_data[image_id][2].append([labels[idx].to(cpu).tolist()])
						answerability_check_data[image_id][3].append([prediction[idx].to(cpu).tolist()])
					
				else:
					answerability_check_data[image_id] = [[question], [[[obj_idx1, obj_idx2]]], [[labels[idx].to(cpu).tolist()]], [[prediction[idx].to(cpu).tolist()]]]
			# ipdb.set_trace()
			# if not labels.size(0) == train_acc.item():
			# 	inds = (labels ==prediction).nonzero().squeeze()
			# 	ipdb.set_trace()

	ipdb.set_trace()
	# outputs = []
	# for idx, input in enumerate(inputs):
	# 	output = inference_demo(model, classifier_head, input)
	# 	outputs.extend(output)

	return  

def get_subset(dict_):
	new_dict = {}
	total_sample = len(dict_.keys())
	subset_idx = random.sample(range(total_sample), int(total_sample*0.1))
	for idx in subset_idx:
		key_ = list(dict_.keys())[idx]
		new_dict[key_] = dict_[key_]

	return new_dict

if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='LBAagent-project')
	#parser.add_argument('--v_sg_path', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/gqa_data/train_sceneGraphs.json')

	# parser.add_argument('--train_scenegraph', type=str, default='/mnt/hdd/hsyoon/workspace/LBA-ARVQA/gqa_data/train_sceneGraphs.json')
	# parser.add_argument('--val_scenegraph', type=str, default='/mnt/hdd/hsyoon/workspace/LBA-ARVQA/gqa_data/val_sceneGraphs.json')

	# parser.add_argument('--train_question', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/train_data_fixed.pkl')
	# parser.add_argument('--val_question', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/val_data_fixed.pkl')
	parser.add_argument('--train_scenegraph', type=str, default='gqa_data/train_sceneGraphs.json')
	parser.add_argument('--val_scenegraph', type=str, default='gqa_data/val_sceneGraphs.json')

	# parser.add_argument('--train_question', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/train_data_fixed.pkl')
	# parser.add_argument('--val_question', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/val_data_fixed.pkl')
	parser.add_argument('--model_name', type=str, default='bert-base-uncased', choices=['bert-base-uncased'])


	args = parser.parse_args()
	
	main(args)
