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

    #with open('/mnt/hdd/hsyoon/workspace/OOD/VQRR/object_list.pkl', 'rb') as f:
    #    object_list = pickle.load(f)
    object_list = []
    lemmatizer = WordNetLemmatizer() 
    new_data = {} 
    current_idx = 0

    for train_question_number in train_questions_list[0].keys():
        print(str(current_idx) + ' out of ' + str(len(train_questions_list[0].keys())))
        current_idx += 1
        question_dict = train_questions_list[0][train_question_number]
        question = question_dict['question']
        imageId = question_dict['imageId']
        ipdb.set_trace()
        type_detailed = train_questions_list[0][train_question_number]["types"]["detailed"]
        
        
        if 'exist' in type_detailed: #we skip existential questions.
            continue
    
        indices_of_objects_dict = question_dict["annotations"]["question"]
        indices_of_objects_list = list(indices_of_objects_dict.keys())

        #TODO delete
        #if train_question_number == '16729733':
        wordtokenize_idx = utils.split_idx2wordtokenize_idx(question, indices_of_objects_list)

                    
        #for lst in lsts:

        #new_data_ = [tuple(word_tokenized_question), []]

        
        #TODO comment out. This is only for getting the object lists
        for object_tuple in wordtokenize_idx:
            object_ = object_tuple[0]
            for idx, object__ in enumerate(object_):
                object_[idx] = object__.lower()

            if object_ not in object_list:
                object_list.append(object_)

    save_file = os.path.join('gqa_data', args.save_filename+'.pkl')
    with open(save_file, 'wb') as f:
        pickle.load(object_list, f)
    return  


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='LBAagent-project')
    parser.add_argument('--question_file_type', type=str, default='train_balanced_questions.json', 
                                    choices=['train_balanced_questions.json', 'val_balanced_questions.json'])
    parser.add_argument('--save_filename', type=str, default='object_list')

    args = parser.parse_args()
    
    main(args)
