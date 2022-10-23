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

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def main(args):

    #load questions 
    train_questions_list = []
    for i in range(0, 1):
        questions = utils.load_v_sg(os.path.join('gqa_data', args.question_file_type))
        train_questions_list.append(questions)

    with open(args.object_list, 'rb') as f:
        object_list = pickle.load(f)

    lemmatizer = WordNetLemmatizer() 
    new_data = {} 
    current_idx = 0

    for train_question_number in train_questions_list[0].keys():
        print(str(current_idx) + ' out of ' + str(len(train_questions_list[0].keys())))
        current_idx += 1
        question_dict = train_questions_list[0][train_question_number]
        question = question_dict['question']
        imageId = question_dict['imageId']
        type_detailed = train_questions_list[0][train_question_number]["types"]["detailed"]
        
        if 'exist' in type_detailed: #we skip existential questions.
            continue
        
        if imageId not in list(new_data.keys()):
            new_data[imageId] = []
        
        #type_structural = train_questions_list[0][train_question_number]["types"]["structural"]
        #type_semantic = train_questions_list[0][train_question_number]["types"]["semantic"]

        
        word_tokenized_question = word_tokenize(question)

        
        indices_of_objects_dict = question_dict["annotations"]["question"]
        indices_of_objects_list = list(indices_of_objects_dict.keys())

        #TODO delete
        #if train_question_number == '16729733':
        wordtokenize_idx = utils.split_idx2wordtokenize_idx(question, indices_of_objects_list)

        #create question&label, instance 
        n = len(wordtokenize_idx)
        lsts = list(itertools.product([0, 1], repeat=n))
        
        #if train_question_number == '16729733':
        for lst in lsts: #go through each augmentation case

            #making the 'red' list (refer to reserach notebook for what 'red' list is)
            new_word_tokenized_question = copy.deepcopy(word_tokenized_question)
            new_wordtokenize_idx = copy.deepcopy(wordtokenize_idx)

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


            #if new_word_tokenized_question == ['Does', 'the', 'brown', 'cross', 'have', 'a', 'different', 'color', 'than', 'the', 'formal', 'woman', 'garbage', 'can', '?']:
            #    ipdb.set_trace()
            new_data_instance = [new_word_tokenized_question, new_wordtokenize_idx,list(lst)]
            new_data[imageId].append(new_data_instance)
                    
        #for lst in lsts:

        #new_data_ = [tuple(word_tokenized_question), []]
        
    # ipdb.set_trace()
    save_file = os.path.join('gqa_data', args.save_filename + '.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(new_data, f)
    return  


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='LBAagent-project')
    # parser.add_argument('--question_path', type=str, default='/mnt/hdd/hsyoon/workspace/OOD/VQRR/gqa_data/')

    parser.add_argument('--model_name', type=str, default='bert-base-uncased', choices=['bert-base-uncased'])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--bsz', type=int, default=2) 

    parser.add_argument('--object_list', type=str, default='gqa_data/object_list.pkl')
    parser.add_argument('--question_file_type', type=str, default='train_balanced_questions.json', 
                                    choices=['train_balanced_questions.json', 'val_balanced_questions.json'])

    parser.add_argument('--save_filename', type=str, default='created_data')

    args = parser.parse_args()
    
    main(args)
