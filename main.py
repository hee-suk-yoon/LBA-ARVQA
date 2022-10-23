import json
import argparse
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
	try:
		os.makedirs(args.tensorboards_path)
		print("Directory " , args.tensorboards_path ,  " Created ") 
	except FileExistsError:
		print("Directory " , args.tensorboards_path ,  " already exists")

	writer = SummaryWriter(args.tensorboards_path)    

	model, tokenizer = utils.get_model(args.model_name)
	model.to(device)

	classifier_head = utils.Predicate2Bool(model.config.hidden_size)
	classifier_head.to(device)

	train_scenegraphs = utils.load_json(args.train_scenegraph)
	val_scenegraphs = utils.load_json(args.val_scenegraph)

	train_questions = utils.load_pickle(args.train_question)
	val_questions = utils.load_pickle(args.val_question)


	# train_scenegraph_ids = list(train_scenegraphs.keys())
	# train_sg2sentence = {train_scenegraph_id : utils.sg2sentence(train_scenegraphs[train_scenegraph_id]) for train_scenegraph_id in train_scenegraph_ids} #convert frame scene graph into a natural language sentence
	# train_sg2sentence_subset = get_subset(train_sg2sentence)
	# tokenized_train_sg2sentence = utils.sentence2tokenize(args, tokenizer, train_sg2sentence_subset)
	# new_train_questions = utils.preprocess_question(args, train_questions)
	# inputs = utils.input_preprocess_V2(args, tokenizer, new_train_questions, tokenized_train_sg2sentence)
	# ipdb.set_trace()
	with open('/mnt/hdd/hsyoon/workspace/OOD/VQRR/processed_data_fixed.pkl', 'rb') as f:
	  inputs = pickle.load(f)

	#train/dev/test split
	total_batch_size = len(inputs)
	train_batch_size = int(total_batch_size*0.8)
	val_batch_size = int((total_batch_size - train_batch_size)/2)
	test_batch_size = total_batch_size-train_batch_size-val_batch_size

	train_data = inputs[:train_batch_size]
	val_data = inputs[train_batch_size:train_batch_size+val_batch_size]
	test_data = inputs[train_batch_size+val_batch_size:]



	loss_criterion = torch.nn.CrossEntropyLoss()
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

	scaler = torch.cuda.amp.GradScaler()
	for epoch in tqdm(range(args.epochs)):
		print(epoch)

		model.train()
		classifier_head.train()
		total_train_loss = 0
		total_val_loss = 0
		for forward_step, batch in enumerate(tqdm(train_data)):
			#print(str(forward_step) + ' out of ' + str(len(train_data)))
			#ipdb.set_trace()
			with torch.cuda.amp.autocast():
				optimizer_PLM.zero_grad()
				optimizer_classifier.zero_grad()
				outputs = model(batch[0].to(device), attention_mask=batch[1].to(device), token_type_ids=batch[2].to(device))['last_hidden_state']
				#get target word features
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
				labels = labels.to(device)
				loss = loss_criterion(output_classifier, labels)
				total_train_loss += loss.item()
				#if not loss:
			#ipdb.set_trace()
			if math.isnan(loss.item()):
				ipdb.set_trace()
			#ipdb.set_trace()
			scaler.scale(loss).backward()


			scaler.unscale_(optimizer_PLM)
			scaler.unscale_(optimizer_classifier)	

			# Since the gradients of optimizer's assigned params are unscaled, clips as usual:
			torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
			torch.nn.utils.clip_grad_norm_(classifier_head.parameters(),args.max_grad_norm)

			scaler.step(optimizer_PLM)
			scaler.step(optimizer_classifier)
			scaler.update()
			#writer.add_scalars('loss', {'train': loss.item()},epoch)
		total_train_loss = total_train_loss/len(train_data) 
		#writer.add_scalars('loss', {'train': loss.item()},epoch)
		with torch.no_grad(): 
			model.eval()
			classifier_head.eval()		
			for forward_step_val, batch_val in enumerate(val_data):
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

				labels = batch_val[4]
				labels = labels.to(device)
				loss = loss_criterion(output_classifier, labels)
				total_val_loss += loss.item()
		total_val_loss = total_val_loss/len(val_data)

		writer.add_scalars('loss', {'train': total_train_loss, 'val': total_val_loss},epoch)
		torch.save(model.state_dict(), os.path.join(args.tensorboards_path, 'bert' + str(epoch) + '.pt'))
		torch.save(classifier_head.state_dict(), os.path.join(args.tensorboards_path, 'classifier' + str(epoch) + '.pt'))
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

	parser.add_argument('--train_scenegraph', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/gqa_data/train_sceneGraphs.json')
	parser.add_argument('--val_scenegraph', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/gqa_data/val_sceneGraphs.json')

	parser.add_argument('--train_question', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/train_data_fixed.pkl')
	parser.add_argument('--val_question', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/val_data.pkl')

	parser.add_argument('--model_name', type=str, default='bert-base-uncased', choices=['bert-base-uncased'])
	parser.add_argument('--max_length', type=int, default=512)
	parser.add_argument('--bsz', type=int, default=64) 
	parser.add_argument('--epochs', type=int, default=100)
	
	parser.add_argument('--tensorboards_path', type=str, default='tensorboard/debug')
	parser.add_argument('--lr_PLM', type=float, default=1e-6)
	parser.add_argument('--lr_classifier', type=float, default=1e-3)
	parser.add_argument('--max_grad_norm', type=float, default=1.0)
	parser.add_argument('--weight_decay', type=float, default=0.01)


	args = parser.parse_args()
	
	main(args)
