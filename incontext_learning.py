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
from transformers import pipeline
#for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")



def main(args):
	generator = pipeline('text-generation', model="facebook/opt-iml-1.3b")

	generator("What is the capital of USA?")
	ipdb.set_trace()
	return

if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='LBAagent-project')
	#parser.add_argument('--v_sg_path', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/gqa_data/train_sceneGraphs.json')

	args = parser.parse_args()
	
	main(args)
	
