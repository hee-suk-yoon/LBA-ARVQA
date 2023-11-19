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
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
#for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")



def main(args):
	#generator = pipeline('text-generation', model="facebook/opt-iml-1.3b")
	model = AutoModelForCausalLM.from_pretrained(args.model_dir,
												torch_dtype=torch.float16,
												).cuda()
	tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

	test_dataloader = utils.get_dataset_fn(args, dataset_type=args.data_dir, tokenizer=tokenizer, split='validation')

	for idx, batch in enumerate(test_dataloader):
		data_clean = batch[0]
		data_augmented = batch[1]

		data_clean_input_ids = data_clean['input_ids'].to(device)
		data_clean_attention_mask = data_clean['attention_mask'].to(device)

		data_augmented_input_ids = data_augmented['input_ids'].to(device)
		data_augmented_attention_mask = data_augmented['attention_mask'].to(device)
		
		output_clean_list = []
		output_aug_list = []
		for i in range(0,1):
			output_clean = model.generate(input_ids=data_clean_input_ids, attention_mask=data_clean_attention_mask, max_length=data_clean_input_ids.shape[1] + 100, num_beams=5, do_sample=True, temperature=0.5, top_p=1.0)
			output_aug = model.generate(input_ids=data_augmented_input_ids, attention_mask=data_augmented_attention_mask, max_length=data_clean_input_ids.shape[1] + 100, num_beams=5, do_sample=True, temperature=0.5, top_p=1.0)


			output_clean_list.append(output_clean)
			output_aug_list.append(output_aug)
		
		loss_clean = []
		loss_aug = []
		lm_embeddings = model.get_input_embeddings()
   		# generated_embedding_vectors has shape [len(opt_embeddings), hidden_size]

		with torch.no_grad():
			for output_clean_instance in output_clean_list:
				generated_embedding_vectors_clean = lm_embeddings(output_clean_instance)

				label_clean = output_clean_instance.clone()
				label_clean[:, :data_clean_input_ids.shape[1]] = -100
				output = model(inputs_embeds=generated_embedding_vectors_clean, labels=label_clean, return_dict=True)

				loss_clean.append(output.loss.item())

		for output_aug_instance in output_aug_list:
			generated_embedding_vectors_aug = lm_embeddings(output_aug_instance)
			generated_embedding_vectors_aug = generated_embedding_vectors_aug.detach().clone()
			generated_embedding_vectors_aug.requires_grad = True

			label_aug = output_aug_instance.clone()
			label_aug[:, :data_augmented_input_ids.shape[1]] = -100
			output = model(inputs_embeds=generated_embedding_vectors_aug, labels=label_aug, return_dict=True)

			loss_aug.append(output.loss.item())
		output.loss.backward()

		#find the quesion index from data_augmented_input_ids
		loc_q_start = torch.where(data_augmented_input_ids==35)[1][-2] + 1
		loc_q_end = torch.where(data_augmented_input_ids==35)[1][-1] -2

		saliency = generated_embedding_vectors_aug.grad[0,loc_q_start:loc_q_end,:]

		#find the L2 norm of the saliency at each time step
		saliency_norm = torch.norm(saliency, dim=1)
		#find the time step with the highest L2 norm
		max_saliency_norm = torch.argmax(saliency_norm)

		#find the word index of the time step with the highest L2 norm
		max_saliency_word = data_augmented_input_ids[0, max_saliency_norm + loc_q_start]

		print('original question: ' + tokenizer.decode(output_clean_list[0][0]))

		print('unanswerable question: '+  tokenizer.decode(output_aug_list[0][0]))

		print('reason: ' + tokenizer.decode(max_saliency_word))
		model.zero_grad()
		ipdb.set_trace()
	return

def get_model(args):
	if args.model_dir == 'facebook/opt-iml-1.3b':
		model = AutoModelForCausalLM.from_pretrained(args.model_dir,
												torch_dtype=torch.float16,
												).cuda()
	return model

if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='LBAagent-project')
	parser.add_argument('--model_dir', type=str, default='facebook/opt-iml-1.3b', help='model directory')
	parser.add_argument('--data_dir', type=str, default='coqa', help='data directory')
	parser.add_argument('--bsz', type=int, default=1, help='batch size')
	args = parser.parse_args()
	
	main(args)
	
