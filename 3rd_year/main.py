import json
import argparse
# from this import d
import torch
import ipdb
import utils
import utils_sys
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from torch import nn, optim
import math
import itertools
import copy

#for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


def process_raw_sentence(exception_object=[], sentences=['This is an example.'], object_list=None, generate_unanswerable_que=False):
    new_data=[]
    if not generate_unanswerable_que:
        for question in sentences:
            # question = sentence

            extracted_objects_ = []
            

            word_tokenized_question = word_tokenize(question)
            try:
                extracted_objects = utils.object_extract(question)
            except:
                ipdb.set_trace()

            for object_, [object_idx1, object_idx2] in extracted_objects:
            # for object_item in extracted_objects:
                exception_list = [1 if exception in object_ else 0 for exception in exception_object]
                if not sum(exception_list) > 0:
                    extracted_objects_.append([[object_], [object_idx1, object_idx2]])

            if len(extracted_objects_) == 0:
                continue
            new_data.append([word_tokenized_question, extracted_objects_])
        return new_data
        # if not generate_unanswerable_que:
        # 	return [word_tokenized_question, extracted_objects_]
    else:
        for question in sentences:
            extracted_objects_ = []

            word_tokenized_question = word_tokenize(question)

            extracted_objects = utils.object_extract(question)

            for object_, [object_idx1, object_idx2] in extracted_objects:
            # for object_item in extracted_objects:
                exception_list = [1 if exception in object_ else 0 for exception in exception_object]
                if not sum(exception_list) > 0:
                    extracted_objects_.append([[object_], [object_idx1, object_idx2]])
            if len(extracted_objects_) == 0:
                continue

            n = len(extracted_objects_)
            lsts = list(itertools.product([0, 1], repeat=n))

            # if question_dict['qid'] not in list(new_data.keys()):
            # 	new_data[question_dict['qid']] = []
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
                        
                        object_ = wordtokenize_idx_single[0]
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
                new_data_instance = [new_word_tokenized_question, new_wordtokenize_idx, list(lst)]
                new_data.append(new_data_instance)
        return new_data

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
    model, tokenizer = utils.get_model(args)
    model.load_state_dict(torch.load(args.model_ckpt))
    model.to(device)

    classifier_head = utils.Predicate2Bool(model.config.hidden_size)
    classifier_head.load_state_dict(torch.load(args.classifier_ckpt))
    classifier_head.to(device)

    # generate output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isfile(os.path.join(args.output_dir, 'output_KAIST.json')):
        output_list = []
    else:
        output_list = utils_sys.read_json(os.path.join(args.output_dir, 'output_KAIST.json'))

    test_questions = utils_sys.read_json(os.path.join(args.dataset_dir, args.test_question))
    sg_fpath = os.path.join(args.dataset_dir, 'AnotherMissOh', 'scene_graph')

    exception_object = ['Haeyoung', 'Jinsang', 'Taejin', 'relationship', 'kind', 'something', 'communication', 'someone', 'everything', 'com', 'color']
    object_list=utils_sys.read_pkl(os.path.join(args.root_dir, args.object_list))

    # for debug, we are currently input scene graph from AnotherMissOh. Maybe the caption of the video can be inputed instead
    for idx, question in enumerate(tqdm(test_questions)):
        output_dict = {}
        test_sg2sentence = {}

        que_idx = idx
        test_sg = utils.AnotherMissOh_sg(args, sg_fpath, question)
        test_sg2sentence[question['qid']] = ". ".join(test_sg)

        output_dict['qid'] = question['qid']
        output_dict['question'] = question['que']
        output_dict['vid'] = question['vid']


        # tokenize the sentenced scene graph
        tokenized_sg2sentence = utils.sentence2tokenize(args, tokenizer, test_sg2sentence)

        # you can input user given sentence like "this is an exmaple"
        # if you input generate_unanswerable_que=True, it will generate unanwerable by folloing the object list. 
        # for debug, we utilize AnotherMissOh obejct list as a dummy


        # for fast test
        test_question = test_questions[que_idx]['que']
        test_qid = test_questions[que_idx]['qid']

        
        # data_instance = process_raw_sentence(exception_object=exception_object, sentences=[test_question], object_list=object_list, generate_unanswerable_que=False) 
        data_instance = process_raw_sentence(exception_object=exception_object, sentences=[test_question], object_list=object_list, generate_unanswerable_que=args.generate_unanswerable_que) 
        # For the case of using user given input question
        # data_instance = process_raw_sentence(exception_object=[], sentences=['This is an example.', 'I have a cat'], object_list=None, generate_unanswerable_que=False) 


        # For the case when generating unanswerable question at the same time with AnotherMissOh
        # data_instance = process_raw_sentence(exception_object=exception_object, sentences=['This is an example.'], object_list=object_list, generate_unanswerable_que=True) 
        inputs = utils.input_preprocess_demo(args, tokenizer, data_instance, tokenized_sg2sentence[test_qid])
        with torch.no_grad():
            for idx, item in enumerate(inputs):
                # label = [[batch[-2], batch[-1].bool()] for batch in item ]
                prediction = inference_demo(model, classifier_head, item)
                answerability = []
                if not prediction:
                    ipdb.set_trace() #TODO
                    answerability.append('anwerable')
                else:
                    for idx_, pred in enumerate(prediction):
                        answerability = []
                        if False in pred.values():
                            answerability.append('unanwerable')
                        else:
                            answerability.append('anwerable')
                        output_dict['answerability'] = answerability
                        output_dict['prediction'] = [prediction[idx_]]
                        output_dict['question'] = " ".join(data_instance[idx_][0])
                        output_dict['GT answerability'] = bool(data_instance[idx_][-1][0])
                        output_list.append(output_dict.copy())
            
            # save the json list 
            utils_sys.write_json(output_list, os.path.join(args.output_dir, args.output_fname))
    return
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='LBAagent-project')

    parser.add_argument('--bbox_threshold', type=float, default=0.3)
    parser.add_argument('--rel_threshold', type=float, default=0.7)

    parser.add_argument('--model_name', type=str, default='bert-base-uncased', choices=['bert-base-uncased'])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--bsz', type=int, default=4) 
    parser.add_argument('--sg_rels_topk', type=int, default=50)
    parser.add_argument('--damping_length', type=int, default=20)

    parser.add_argument('--root_dir', type=str, default='/mnt/hsyoon/workspace/LBA-ARVQA/3rd_year')
    
    parser.add_argument('--model_ckpt', type=str, default='./ckpt/2023-10-30_15:13:30_best_loss.pt')
    parser.add_argument('--classifier_ckpt', type=str, default='./ckpt/2023-10-30_15:13:30_classifier_best_loss.pt')
    parser.add_argument('--object_list', type=str, default='./saves/AnotherMissOhQA_object_list.pkl')
    parser.add_argument('--test_question', type=str, default='DramaQA/AnotherMissOhQA_test_set.json')

    parser.add_argument('--dataset_dir', type=str, default='../dataset')
    parser.add_argument('--output_dir', type=str, default='./LBA_2024')
    parser.add_argument('--output_fname', type=str, default='output_KAIST.json')

    parser.add_argument('--generate_unanswerable_que', action='store_true')
    args = parser.parse_args()
    
    main(args)