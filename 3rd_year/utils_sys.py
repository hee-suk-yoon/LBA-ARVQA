import torch
import os
import pickle
import json
import ipdb
import torch.distributed as dist
import csv
import string
import numpy as np
import random

from typing import Tuple, Union, Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass

def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()

def is_main_process():
	return get_rank() == 0

def init_distributed_mode(args):
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		args.rank = int(os.environ["RANK"])
		args.world_size = int(os.environ['WORLD_SIZE'])
		args.gpu = int(os.environ['LOCAL_RANK'])
		
		# os.environ["MASTER_ADDR"] = '143.248.159.12'
		# os.environ['MASTER_PORT'] = '8889'

	elif 'SLURM_PROCID' in os.environ:
		args.rank = int(os.environ['SLURM_PROCID'])
		args.gpu = args.rank % torch.cuda.device_count()
	else:
		print('Not using distributed mode')
		args.distributed = False
		return args

	args.distributed = True

	torch.cuda.set_device(args.gpu)
	args.dist_backend = 'nccl'
	
	print('| distributed init (rank {}): {}'.format(
		args.rank, args.dist_url), flush=True)
	torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
										 world_size=args.world_size, rank=args.rank)
	torch.distributed.barrier()
	
	return args

def empty_dataset(keys):
	empty_dataset = {key:[] for key in keys}
	return empty_dataset

def merging_dict(dict1, dict2):
	new_dict = defaultdict(list)
	for d in (dict1, dict2):
		for key, value in d.items():
			new_dict[key] += d[key]
	return new_dict

def dict_to_device(dict, device):
	for key in dict.keys():
		dict[key] = dict[key].to(device)
	return dict

def dict_to_devices(dict, rank):
    for key in dict.keys():
        dict[key] = dict[key].cuda(rank)
    return dict


def save_pkl(data, file):
	with open(file, 'wb') as f:
		pickle.dump(data, f)
	return

def read_pkl(file):
	with open(file, 'rb') as f:
		data = pickle.load(f)
	return data

def append_pkl(data, file):
	with open(file, 'ab') as f:
		pickle.dump(data, f)
	return

def read_txt(file):
	file = open(file, 'r')
	data = []
	while True:
		line = file.readline()
		data.append(line.strip().split('\t'))
		if not line:
			break
	file.close()

	return data

def write_csv(file, data, column_name):
	with open(file, 'w', newline='\n') as f:
		write = csv.writer(f)
		assert len(data) == len(column_name)
		write.writerow(column_name)
		for i in range(len(data[0])):
			line = [data[column_idx][i] for column_idx, _ in enumerate(column_name)]
			write.writerow(line)
	return

def append_csv(file, data):
	with open(file, 'a', newline='\n') as f:
		write = csv.writer(f)
		for i in range(len(data[0])):
			line = [data[j][i] for j in range(len(data))]
			write.writerow(line)
	return

def read_csv(file):
	data = []
	alphabet = list(string.ascii_letters)
	with open(file, 'r', newline='') as f:
		reader = csv.reader(f, delimiter=' ',  quotechar='|')
		for row in reader:
			# ipdb.set_trace()
			if len([i for i in row[0] if i not in alphabet]) == 0:
				data.append(row[0])
	return data

def read_jsonl(file):
	with open(file, 'r', encoding="utf-8") as f:
		data = [json.loads(line) for line in f]
	return data

def read_json(path):
    f = open(path)
    data = json.load(f)  
    return data  

def write_json(data, path):
	with open(path, 'w') as f:
		json.dump(data, f, ensure_ascii=False, indent=4)
	return

def set_run_name(args):
	from datetime import date, datetime, timezone, timedelta
	KST  = timezone(timedelta(hours=9))
	date = str(date.today())
	time_record = str(datetime.now(KST).time())[:8]

	args.run_name = date+'_'+time_record

	return args

def set_gpu(args):
	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.cpu = torch.device("cpu")
	args = init_distributed_mode(args)
	return args

def set_seed(args, seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic=True

	if args.distributed:
		torch.cuda.manual_seed_all(args.seed)
	return


def get_subset(dict_):
	new_dict = {}
	total_sample = len(dict_.keys())
	subset_idx = random.sample(range(total_sample), int(total_sample*0.05))
	for idx in subset_idx:
		key_ = list(dict_.keys())[idx]
		new_dict[key_] = dict_[key_]

	return new_dict
