import json 
import torch 
from nltk.tokenize import word_tokenize
import ipdb
import pickle
import random 
import os
import glob
import copy
from dataclasses import dataclass
from torch.utils.data import DataLoader
import spacy

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from random import choice

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_json(path):
    f = open(path)
    data = json.load(f)  
    return data  


def sg2sentence(sg):
    list_of_objects = list(sg['objects'].keys())
    dict_object_count = {el:0 for el in list_of_objects}  #This is for those nodes (with potential attribute) without any outgoing or incoming edge (i.e., isolated.)
    
    def get_name_and_attribute(dict_):
        object_name = dict_['name']
        list_object_attribute = dict_['attributes']
        return object_name, list_object_attribute

    generated_sentences = []
    for idx, object_ in enumerate(list_of_objects):
        object_name, list_object_attribute = get_name_and_attribute(sg['objects'][object_])

        list_of_relations = sg['objects'][object_]['relations']
        if list_of_relations == []:
            continue
        
        else:
            for idx2, relation in enumerate(list_of_relations):
                target_object = sg['objects'][relation['object']]
                target_object_name, list_target_object_attribute = get_name_and_attribute(target_object)
                relation_name = relation['name']

                
                #generate sentence
                oa_string = ", ".join(map(str, list_object_attribute)) #object attribute string
                if oa_string != '':
                    oa_o_string = oa_string + ' ' +object_name # object attribute + object string 
                else:
                    oa_o_string = object_name
                
                toa_string = ", ".join(map(str, list_target_object_attribute)) #target object attribute string 
                if toa_string != '':
                    toa_o_string = toa_string + ' ' +target_object_name # object attribute + object string 
                else:
                    toa_o_string = target_object_name

                oa_o_r_to_string = oa_o_string + ' ' + relation_name + ' ' + toa_o_string
                generated_sentences.append(oa_o_r_to_string)
                dict_object_count[object_] += 1
                dict_object_count[relation['object']] += 1
                
                #for object_attribute in enumerate(list_object_attribute):
                #    #object_attribute_string += object_attribute + ', '
                #    ", ".join(map(str, alist))


    #isolated nodes
    for idx, object_ in enumerate(list(dict_object_count.keys())):
        if dict_object_count[object_] == 0:
            object_name, list_object_attribute = get_name_and_attribute(sg['objects'][object_])

            #generate sentence
            oa_string = ", ".join(map(str, list_object_attribute)) #object attribute string
            if oa_string != '':
                oa_o_string = oa_string + ' ' +object_name # object attribute + object string 
            else:
                oa_o_string = object_name
            
            generated_sentences.append(oa_o_string)
            dict_object_count[object_] += 1
    
    return ". ".join(map(str, generated_sentences))

def customsg2sentence(info, sg, topk):
    file_ids = ['/'.join(file.split('/')[-4:]) for file in info['idx_to_files']]

    classes = info['ind_to_classes']
    predicates = info['ind_to_predicates']
    sg_keys = list(sg.keys())
    sg2sent_dict = {}
    for idx, sg_key in enumerate(sg_keys):
        rel_labels = [predicates[i] for i in sg[sg_key]['rel_labels']]
        rel_pairs = [[classes[i] for i in sg[sg_key]['rel_pairs'][j]] for j in range(len(sg[sg_key]['rel_pairs']))]
        # rels = copy.deepcopy(rel_pairs)
        rels = []
        for idx_, item in enumerate(rel_pairs):
            if item[0] == '__background__':
                rels.append(['there', 'is', item[1]])
            if '__background__' not in item:
                rels.append([item[0], rel_labels[idx_], item[1]])
            
            if item [1] == '__background__':
                continue

        rels = rels[:topk]        

        sentences = [' '.join(sent) + '.' for sent in rels]
        sg2sent_dict[file_ids[idx]] = " ".join(sentences)
    
    return sg2sent_dict



import sng_parser #https://github.com/vacancy/SceneGraphParser
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

def object_extract(question):
    wordsList = nltk.word_tokenize(question)
    scene_graph = sng_parser.parse(question) #I am not sure but sng_parser seems to use nltk.word_tokenize. (can't detect woman in "the woman's shoes")

    objects = []
    for entity in scene_graph['entities']:
        span = entity['span']
        span_bounds = entity['span_bounds']
        head = entity['head']
        words = nltk.word_tokenize(span)
        if head in words:
            pos = words.index(head)
        span_bounds_head = (span_bounds[0]+pos,span_bounds[0]+pos+1)
        objects.append((head,span_bounds_head))

    #include only nouns for the objects
    tagged = nltk.pos_tag(wordsList)

    objects_passed = [] #objects that have passed the Noun POS test
    for idx, (head, span_bounds_head) in enumerate(objects):
        tagged_idx = span_bounds_head[0]
        pos = tagged[tagged_idx][1]
        if pos in ['NN', 'NNS']:
            objects_passed.append((head, span_bounds_head))

    return objects_passed



from transformers import BertTokenizer, BertModel

def get_model(args):
    # if model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name, max_length=args.max_length)

    return model, tokenizer

def split_idx2wordtokenize_idx(question, split_idx):
    #ipdb.set_trace()
    #question_without_last_punctuation = question[:-1]
    #question_without_last_punctuation_split = question_without_last_punctuation.split(' ')
    question_split = question.split(' ')
    word_tokenized_question = word_tokenize(question)
    wordtokenize_idx = []
    for split_idx_string in split_idx:
        split_idx_ = split_idx_string.split(':')

        #get what the target word is, and its word tokenized form.
        if len(split_idx_) == 1: #single word split idx
            target_word = question_split[int(split_idx_[0])]
            target_word_tokenized = word_tokenize(target_word)
            if len(target_word_tokenized) >= 3:
                ipdb.set_trace() #just to make sure 
            target_word_tokenized = [target_word_tokenized[0]] #get the first to remove punctuations etc. (e.g. ',' ''s' '?')


        else: #multiple word (i.e., '?:?')
            target_word = question_split[int(split_idx_[0]):int(split_idx_[1])]
            length_split = len(target_word)
            #target_word_tokenized = word_tokenize(target_word)
            #target_word_tokenized = [t for t_ in target_word]
            target_word_tokenized = []
            for t_ in target_word:
                target_word_tokenized.extend(word_tokenize(t_))

            if '\'s' in target_word_tokenized: #''s' need to consider... (made some assumptions. Might need to change this)
                index_s = target_word_tokenized.index('\'s')
                target_word_tokenized = target_word_tokenized[index_s-1:index_s+2]


            elif len(target_word_tokenized) - length_split >= 2:
                ipdb.set_trace() #just to make sure 

            elif len(target_word_tokenized) != length_split:
                target_word_tokenized = target_word_tokenized[:-1]

        #for idx, target_word_tokenized_single in enumerate(target_word_tokenized):
        #    wordtokenized_idx = word_tokenized_question.index(target_word_tokenized_single)   
        

        """ #TODO had error so replacing.
        wordtokenized_start_idx = word_tokenized_question.index(target_word_tokenized[0])
        wordtokenized_end_idx = word_tokenized_question.index(target_word_tokenized[-1])
        wordtokenize_idx.append([target_word_tokenized, [wordtokenized_start_idx,wordtokenized_end_idx+1]])
        """

        word_index = getIndex(word_tokenized_question, target_word_tokenized)
        if word_index == None: 
            ipdb.set_trace()
        wordtokenized_start_idx = word_index[0]
        if word_index[0] == word_index[1]: #single word
            wordtokenized_end_idx = wordtokenized_start_idx + 1
        else: 
            wordtokenized_end_idx = word_index[1]
        wordtokenize_idx.append([target_word_tokenized, [wordtokenized_start_idx,wordtokenized_end_idx]])
        
    return wordtokenize_idx

def getIndex(lst1, lst2):
    index1 = next((i for i in range(len(lst1) - len(lst2)) if lst1[i:i + len(lst2)] == lst2), None)
    if index1 is not None: # when sub list doesn't exist in another list
        return (index1, index1) if len(lst2) == 1 else (index1, index1+len(lst2))
    else:
        return None

def preprocess_question(args, questions):
    new_data_v2 = {}
    for imageid in questions.keys():
        new_data_v2[imageid] = []

    for imageid in questions.keys():
        list_question_data = questions[imageid]
        for single_question_data in list_question_data: #red 
            word_tokenized_question = single_question_data[0]
            # if word_tokenized_question == ['What', 'type', 'of', 'highway', 'is', 'to', 'the', 'right', 'of', 'the', 'gray', 'fence', 'green', 'vegetable', '?']:
            #     ipdb.set_trace()
            object_indices = single_question_data[1] #blue
            for idx, object_index in enumerate(object_indices): #green
                new_single_question_data = [word_tokenized_question, [object_index], [single_question_data[2][idx]]] #new red
                new_data_v2[imageid].append(new_single_question_data)        

    return new_data_v2

def input_preprocess_V2(args, tokenizer, questions, frame_sg2sentence):
    inputs_ids = []
    inputs_attn = []
    inputs_type_ids = []
    output_idx_all = []
    labels_all = []
    #for frame_question_pair in list_frame_question_pairs:
    total_length = 0
    for imageid in list(questions.keys()):
    #for imageid in list(frame_sg2sentence.keys()):
        if imageid in list(frame_sg2sentence.keys()):
            for frame_question_pair in questions[imageid]:
                total_length+=1
    
    current_idx = 0
    for imageid in list(questions.keys()):
        if imageid in list(frame_sg2sentence.keys()):
    #for imageid in list(frame_sg2sentence.keys()):
        #frame_sentence = frame_sg2sentence[imageid]
            for frame_question_pair in questions[imageid]: #one red
                current_idx += 1
                print(str(current_idx) + 'out of ' + str(total_length))
                #question = frame_question_pair[1]
                #predicates = object_extract(question)
                
                #frame_sentence_split = word_tokenize(frame_sentence)
                question_split = frame_question_pair[0]
                predicates = frame_question_pair[1]
                if predicates == []:
                    continue
                #ipdb.set_trace()
                output_idx = {str(el[0]):[] for el in predicates}
                #output_idx = [[] for el in predicates]
                #if len(predicates) != len(output_idx.keys()): #두개의 object가 같을때 있을수 있음.
                #    continue

                #label 
                label_ = frame_question_pair[2]
                #ipdb.set_trace()
                #tokenize and tensorize data
                c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)])] #cls token aka sos token, returns a list with index # 
                target_ids = [0]
                attention_m = [1]
                idx_master = 1

                word_ids = frame_sg2sentence[imageid]
                target_ids.extend([0]*len(word_ids))
                attention_m.extend([1]*len(word_ids))
                c_ids.extend(word_ids)
                idx_master += len(word_ids)
                # for idx, word in enumerate(frame_sentence_split):
                #     word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
                #     target_ids.extend([0]*len(word_ids))
                #     attention_m.extend([1]*len(word_ids))
                #     c_ids.extend(word_ids)
                #     idx_master += len(word_ids)


                c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token 
                target_ids.extend([0])
                attention_m.extend([1])
                idx_master += 1
                
                active_pred_idx = -1 
                active_pred = False
                for idx, word in enumerate(question_split):
                    #check if idx falls into any span 
                    for pred_idx, predicate in enumerate(predicates):
                        if predicate[1][0] <= idx and idx < predicate[1][1]:
                            active_pred_idx = pred_idx 
                            active_pred = True
                            break
                        else:
                            active_pred_idx = -1
                            active_pred = False
                    
                    word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
                    target_ids.extend([1]*len(word_ids))
                    attention_m.extend([1]*len(word_ids))
                    c_ids.extend(word_ids)


                    if active_pred:
                        output_idx[str(predicates[active_pred_idx][0])].extend(list(range(idx_master,idx_master+len(word_ids))))

                    idx_master += len(word_ids)
                
                c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token 
                target_ids.extend([1])
                attention_m.extend([1])        
                if output_idx[list(output_idx.keys())[0]] == []:
                    ipdb.set_trace()

                #padding
                if len(attention_m) >= args.max_length:
                    continue
                else:
                    c_ids, attention_m, target_ids = normalize_length(c_ids, attention_m, target_ids, args.max_length, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])

                    inputs_ids.append(torch.cat(c_ids, dim=-1)) 
                    inputs_attn.append(torch.tensor(attention_m).unsqueeze(dim=0)) 
                    inputs_type_ids.append(torch.tensor(target_ids).unsqueeze(dim=0)) 
                    output_idx_all.append(output_idx)
                    labels_all.append(label_)

    data = list(zip(inputs_ids, inputs_attn, inputs_type_ids, output_idx_all,labels_all))

    random.shuffle(data)
    if args.bsz > 1: #
        print('Batching data with bsz={}...'.format(args.bsz)) #
        batched_data = [] # 
        for idx in range(0, len(data), args.bsz): #
            if idx+args.bsz <=len(data): b = data[idx:idx+args.bsz] #
            else: b = data[idx:] #
            context_ids = torch.cat([x for x,_,_,_,_ in b], dim=0) #
            context_attn_mask = torch.cat([x for _,x,_,_,_ in b], dim=0) #
            context_type_id = torch.cat([x for _,_,x,_,_ in b], dim=0) #
            context_output_idx = [x for _,_,_,x,_ in b]
            context_label = torch.tensor([x for _,_,_,_,x in b]).view(-1)
            
            #labels = [] #
            #for _,_,_,_,_,x in b: labels.extend(x) #
            #batched_data.append((context_ids, context_attn_mask, context_type_id, example_keys, instances, labels)) # 
            batched_data.append((context_ids, context_attn_mask, context_type_id, context_output_idx, context_label)) # 
        return batched_data #
    
    else:
        return data


def input_preprocess_V3(args, tokenizer, questions, frame_sg2sentence):
    inputs_ids = []
    inputs_attn = []
    inputs_type_ids = []
    output_idx_all = []
    labels_all = []
    #for frame_question_pair in list_frame_question_pairs:
    total_length = 0
    for imageid in list(questions.keys()):
    #for imageid in list(frame_sg2sentence.keys()):
        if imageid in list(frame_sg2sentence.keys()):
            for frame_question_pair in questions[imageid]:
                total_length+=1
    
    current_idx = 0
    for imageid in list(questions.keys()):
        ipdb.set_trace()
        if imageid in list(frame_sg2sentence.keys()):
    #for imageid in list(frame_sg2sentence.keys()):
        #frame_sentence = frame_sg2sentence[imageid]
            for frame_question_pair in questions[imageid]: #one red
                current_idx += 1
                print(str(current_idx) + ' out of ' + str(total_length))
                #question = frame_question_pair[1]
                #predicates = object_extract(question)
                
                #frame_sentence_split = word_tokenize(frame_sentence)
                question_split = frame_question_pair[0]
                predicates = frame_question_pair[1]
                if predicates == []:
                    continue
                #ipdb.set_trace()
                output_idx = {str(el[0]):[] for el in predicates}
                #output_idx = [[] for el in predicates]
                #if len(predicates) != len(output_idx.keys()): #두개의 object가 같을때 있을수 있음.
                #    continue

                #label 
                label_ = frame_question_pair[2]
                #ipdb.set_trace()
                #tokenize and tensorize data
                c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)])] #cls token aka sos token, returns a list with index # 
                target_ids = [0]
                attention_m = [1]
                idx_master = 1

                word_ids = frame_sg2sentence[imageid]
                target_ids.extend([0]*len(word_ids))
                attention_m.extend([1]*len(word_ids))
                c_ids.extend(word_ids)
                idx_master += len(word_ids)
                # for idx, word in enumerate(frame_sentence_split):
                #     word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
                #     target_ids.extend([0]*len(word_ids))
                #     attention_m.extend([1]*len(word_ids))
                #     c_ids.extend(word_ids)
                #     idx_master += len(word_ids)
                 # added by esyoon 2023-11-01-15:22:04
                if len(c_ids) > args.max_length:
                    c_ids = c_ids[:args.max_length-15-len(question_split)]
                    target_ids = target_ids[:args.max_length-15-len(question_split)]
                    attention_m = attention_m[:args.max_length-15-len(question_split)]
                    idx_master = args.max_length-15-len(question_split)

                c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token 
                target_ids.extend([0])
                attention_m.extend([1])
                idx_master += 1
                
                active_pred_idx = -1 
                active_pred = False
                for idx, word in enumerate(question_split):
                    #check if idx falls into any span 
                    for pred_idx, predicate in enumerate(predicates):
                        if predicate[1][0] <= idx and idx < predicate[1][1]:
                            active_pred_idx = pred_idx 
                            active_pred = True
                            break
                        else:
                            active_pred_idx = -1
                            active_pred = False
                    
                    word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
                    target_ids.extend([1]*len(word_ids))
                    attention_m.extend([1]*len(word_ids))
                    c_ids.extend(word_ids)


                    if active_pred:
                        output_idx[str(predicates[active_pred_idx][0])].extend(list(range(idx_master,idx_master+len(word_ids))))

                    idx_master += len(word_ids)
                
                c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token 
                target_ids.extend([1])
                attention_m.extend([1])        
                if output_idx[list(output_idx.keys())[0]] == []:
                    ipdb.set_trace()
                ipdb.set_trace()
                #padding
                if len(attention_m) >= args.max_length:
                    ipdb.set_trace()
                    continue
                else:
                    c_ids, attention_m, target_ids = normalize_length(c_ids, attention_m, target_ids, args.max_length, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])

                inputs_ids.append(torch.cat(c_ids, dim=-1)) 
                inputs_attn.append(torch.tensor(attention_m).unsqueeze(dim=0)) 
                inputs_type_ids.append(torch.tensor(target_ids).unsqueeze(dim=0)) 
                output_idx_all.append(output_idx)
                labels_all.append(label_)
                ipdb.set_trace()
    data = list(zip(inputs_ids, inputs_attn, inputs_type_ids, output_idx_all,labels_all))
    
    random.shuffle(data)
    if args.bsz > 1: #
        print('Batching data with bsz={}...'.format(args.bsz)) #
        batched_data = [] # 
        for idx in range(0, len(data), args.bsz): #
            if idx+args.bsz <=len(data): b = data[idx:idx+args.bsz] #
            else: b = data[idx:] #
            context_ids = torch.cat([x for x,_,_,_,_ in b], dim=0) #
            context_attn_mask = torch.cat([x for _,x,_,_,_ in b], dim=0) #
            context_type_id = torch.cat([x for _,_,x,_,_ in b], dim=0) #
            context_output_idx = [x for _,_,_,x,_ in b]
            context_label = torch.tensor([x for _,_,_,_,x in b]).view(-1)
            
            #labels = [] #
            #for _,_,_,_,_,x in b: labels.extend(x) #
            #batched_data.append((context_ids, context_attn_mask, context_type_id, example_keys, instances, labels)) # 
            batched_data.append((context_ids, context_attn_mask, context_type_id, context_output_idx, context_label)) # 
        return batched_data #
    
    else:
        return data

def input_preprocess_demo(args, tokenizer, questions, sg2sentence):
    inputs_ids = []
    inputs_attn = []
    inputs_type_ids = []
    output_idx_all = []
    labels_all = []
    for question in questions:
        frame_question_pair = question
        #frame_sentence_split = word_tokenize(frame_sentence)
        question_split = frame_question_pair[0]
        predicates = frame_question_pair[1]
        if predicates == []:
            raise ValueError("there is no object in the given sentence")
        #ipdb.set_trace()
        output_idx = {str(el[0]):[] for el in predicates}
        #output_idx = [[] for el in predicates]
        #if len(predicates) != len(output_idx.keys()): #두개의 object가 같을때 있을수 있음.
        #    continue

        #label 
        # label_ = frame_question_pair[2]
        #ipdb.set_trace()
        #tokenize and tensorize data
        c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)])] #cls token aka sos token, returns a list with index # 
        target_ids = [0]
        attention_m = [1]
        idx_master = 1

        word_ids = sg2sentence
        target_ids.extend([0]*len(word_ids))
        attention_m.extend([1]*len(word_ids))
        c_ids.extend(word_ids)
        idx_master += len(word_ids)

        # for idx, word in enumerate(frame_sentence_split):
        #     word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
        #     target_ids.extend([0]*len(word_ids))
        #     attention_m.extend([1]*len(word_ids))
        #     c_ids.extend(word_ids)
        #     idx_master += len(word_ids)

        # added by esyoon 2023-11-01-15:22:04
        if len(c_ids) > args.max_length:
            c_ids = c_ids[:args.max_length-15-len(question_split)]
            target_ids = target_ids[:args.max_length-15-len(question_split)]
            attention_m = attention_m[:args.max_length-15-len(question_split)]
            idx_master = idx_master - args.max_length-15-len(question_split)

        c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token 
        target_ids.extend([0])
        attention_m.extend([1])
        idx_master += 1
        
        active_pred_idx = -1 
        active_pred = False
        for idx, word in enumerate(question_split):
            #check if idx falls into any span 
            for pred_idx, predicate in enumerate(predicates):
                if predicate[1][0] <= idx and idx < predicate[1][1]:
                    active_pred_idx = pred_idx 
                    active_pred = True
                    break
                else:
                    active_pred_idx = -1
                    active_pred = False
            
            word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
            target_ids.extend([1]*len(word_ids))
            attention_m.extend([1]*len(word_ids))
            c_ids.extend(word_ids)

            if active_pred:
                output_idx[str(predicates[active_pred_idx][0])].extend(list(range(idx_master,idx_master+len(word_ids))))

            idx_master += len(word_ids)
        
        c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token 
        target_ids.extend([1])
        attention_m.extend([1])        
        if output_idx[list(output_idx.keys())[0]] == []:
            ipdb.set_trace()


        #padding
        if len(attention_m) >= args.max_length:
            ipdb.set_trace()
        else:
            c_ids, attention_m, target_ids = normalize_length(c_ids, attention_m, target_ids, args.max_length, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])

            inputs_ids.append(torch.cat(c_ids, dim=-1)) 
            inputs_attn.append(torch.tensor(attention_m).unsqueeze(dim=0)) 
            inputs_type_ids.append(torch.tensor(target_ids).unsqueeze(dim=0)) 
            output_idx_all.append(output_idx)
            # labels_all.append(label_)

    data = list(zip(inputs_ids, inputs_attn, inputs_type_ids, output_idx_all))
    if args.bsz > 1: #
        print('Batching data with bsz={}...'.format(args.bsz)) #
        batched_data = [] # 
        for idx in range(0, len(data), args.bsz): #
            if idx+args.bsz <=len(data): b = data[idx:idx+args.bsz] #
            else: b = data[idx:] #
            context_ids = torch.cat([x for x,_,_,_ in b], dim=0) #
            context_attn_mask = torch.cat([x for _,x,_,_ in b], dim=0) #
            context_type_id = torch.cat([x for _,_,x,_ in b], dim=0) #
            context_output_idx = [x for _,_,_,x in b]
            # context_label = torch.tensor([x for _,_,_,_,x in b]).view(-1)
            
            #labels = [] #
            #for _,_,_,_,_,x in b: labels.extend(x) #
            #batched_data.append((context_ids, context_attn_mask, context_type_id, example_keys, instances, labels)) # 
            # batched_data.append((context_ids, context_attn_mask, context_type_id, context_output_idx, context_label)) # 
            batched_data.append((context_ids, context_attn_mask, context_type_id, context_output_idx)) # 
        return batched_data #
    
    else:
        return data


def sentence2tokenize(args, tokenizer, train_sg2sentence):
    tokenized_train_sg2sentence = {}
    total_length = len(list(train_sg2sentence.keys()))
    current_idx = 0

    
    for train_scenegraph_id in list(train_sg2sentence.keys()):
        current_idx += 1
        print(str(current_idx) + ' out of ' + str(total_length))

        frame_sentence_split = word_tokenize(train_sg2sentence[train_scenegraph_id])
        total_word_ids = []
        for idx, word in enumerate(frame_sentence_split):
            word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
            total_word_ids.extend(word_ids)
        tokenized_train_sg2sentence[train_scenegraph_id] = total_word_ids
    
    return tokenized_train_sg2sentence

def demo_sentence2tokenize(args, tokenizer, train_sg2sentence):

    frame_sentence_split = word_tokenize(train_sg2sentence)
    total_word_ids = []
    for idx, word in enumerate(frame_sentence_split):
        word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
        total_word_ids.extend(word_ids)
    
    return total_word_ids

def input_preprocess(args, tokenizer, list_frame_question_pairs, frame_sg2sentence):
    inputs_ids = []
    inputs_attn = []
    inputs_type_ids = []
    output_idx_all = []

    for frame_question_pair in list_frame_question_pairs:
        frame_sentence = frame_sg2sentence[frame_question_pair[0]]
        question = frame_question_pair[1]
        predicates = object_extract(question)

        frame_sentence_split = word_tokenize(frame_sentence)
        question_split = word_tokenize(question)

        output_idx = {el[0]:[] for el in predicates}
        #tokenize and tensorize data
        c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)])] #cls token aka sos token, returns a list with index # 
        target_ids = [0]
        attention_m = [1]
        idx_master = 1

        for idx, word in enumerate(frame_sentence_split):
            word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
            target_ids.extend([0]*len(word_ids))
            attention_m.extend([1]*len(word_ids))
            c_ids.extend(word_ids)
            idx_master += len(word_ids)


        c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token 
        target_ids.extend([0])
        attention_m.extend([1])

        
        active_pred_idx = -1 
        active_pred = False
        for idx, word in enumerate(question_split):
            #check if idx falls into any span 
            for pred_idx, predicate in enumerate(predicates):
                if predicate[1][0] <= idx and idx < predicate[1][1]:
                    active_pred_idx = pred_idx 
                    active_pred = True
                    break
                else:
                    active_pred_idx = -1
                    active_pred = False
            
            word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word, add_special_tokens=False)]
            target_ids.extend([1]*len(word_ids))
            attention_m.extend([1]*len(word_ids))
            c_ids.extend(word_ids)


            if active_pred:
                output_idx[predicates[active_pred_idx][0]].extend(list(range(idx_master,idx_master+len(word_ids))))

            idx_master += len(word_ids)
        
        c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)])) #aka eos token 
        target_ids.extend([1])
        attention_m.extend([1])        

        #padding
        if len(attention_m) > args.max_length:
            continue
        else:
            c_ids, attention_m, target_ids = normalize_length(c_ids, attention_m, target_ids, args.max_length, pad_id=tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])

            inputs_ids.append(torch.cat(c_ids, dim=-1)) 
            inputs_attn.append(torch.tensor(attention_m).unsqueeze(dim=0)) 
            inputs_type_ids.append(torch.tensor(target_ids).unsqueeze(dim=0)) 
            output_idx_all.append(output_idx)

    data = list(zip(inputs_ids, inputs_attn, inputs_type_ids, output_idx_all))

    if args.bsz > 1: #
        print('Batching data with bsz={}...'.format(args.bsz)) #
        batched_data = [] # 
        for idx in range(0, len(data), args.bsz): #
            if idx+args.bsz <=len(data): b = data[idx:idx+args.bsz] #
            else: b = data[idx:] #
            context_ids = torch.cat([x for x,_,_,_ in b], dim=0) #
            context_attn_mask = torch.cat([x for _,x,_,_ in b], dim=0) #
            context_type_id = torch.cat([x for _,_,x,_ in b], dim=0) #
            context_output_idx = [x for _,_,_,x in b]
            #labels = [] #
            #for _,_,_,_,_,x in b: labels.extend(x) #
            #batched_data.append((context_ids, context_attn_mask, context_type_id, example_keys, instances, labels)) # 
            batched_data.append((context_ids, context_attn_mask, context_type_id, context_output_idx)) # 
        return batched_data #
    
    else:
        return data

def normalize_length(ids, attn_mask, o_mask, max_len, pad_id):
	if max_len == -1: #
		return ids, attn_mask, o_mask #
	else: # 
		if len(ids) < max_len: #
			while len(ids) < max_len: #
				ids.append(torch.tensor([[pad_id]])) # 
				attn_mask.append(0) # 
				o_mask.append(0) #
		else: # 
			ids = ids[:max_len-1]+[ids[-1]] # 
			attn_mask = attn_mask[:max_len] # 
			o_mask = o_mask[:max_len] # 

		assert len(ids) == max_len #
		assert len(attn_mask) == max_len #
		assert len(o_mask) == max_len # 

		return ids, attn_mask, o_mask #

class Predicate2Bool(torch.nn.Module):

    def __init__(self, hidden_feature_length):
        super(Predicate2Bool, self).__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_feature_length, out_features=120),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=84, out_features=2),
        )


    def forward(self, x):
        logits = self.classifier(x)
        return logits


def dummy_AnotherMissOh_QA(question_data):
    dummy_vids = ['AnotherMissOh01_001_0078']
    dummy_data_dict = {dummy_vid:[] for dummy_vid in dummy_vids}

    for idx, data in enumerate(question_data):
        if data['vid'] in dummy_vids:
            dummy_data_dict[data['vid']].append(data)
    return dummy_data_dict

def AnotherMissOh_sg(args, sg_fpath, question):

    if question['videoType'] == 'shot':
        sg_fname = '/'.join(question['vid'].split('_'))
        
        try:
            data_info = load_json(os.path.join(sg_fpath, sg_fname, 'custom_data_info.json'))
              
        
        except:
            # from 12 ~ there is no custom_info.json so we are using 01/001/0078
            data_info = load_json(os.path.join(sg_fpath, 'AnotherMissOh01/001/0078', 'custom_data_info.json'))
       
        idx2node = data_info['ind_to_classes']             
        idx2relation = data_info['ind_to_predicates'] 

        sg_data = load_json(os.path.join(sg_fpath, sg_fname, 'custom_prediction.json'))         

        phrases = []

        for k, sg_item in sg_data.items():

            for idx, relation in enumerate(sg_item['rel_pairs']):
                node1_idx, node2_idx = relation

                node1 = idx2node[sg_item['bbox_labels'][node1_idx]]
                node2 = idx2node[sg_item['bbox_labels'][node2_idx]]

                node1_score = sg_item['bbox_scores'][node1_idx]
                node2_score = sg_item['bbox_scores'][node2_idx]
                relation = idx2relation[sg_item['rel_labels'][idx]]
                relation_score = sg_item['rel_scores'][idx]

                if (node1_score> args.bbox_threshold) and (node2_score> args.bbox_threshold) and (relation_score> args.rel_threshold):
                    phrases.append(node1 + ' '+ relation + ' ' + node2)

    
    elif question['videoType'] == 'scene':
        sg_fnames = '/'.join(question['vid'].split('_')[:-1])
        sg_files = glob.glob(os.path.join(sg_fpath, sg_fnames, '*'))
        phrases = []

        for sg_file in sg_files:
            try:
                data_info = load_json(os.path.join(sg_fpath, sg_fname, 'custom_data_info.json'))
            except:
                # from 12 ~ there is no custom_info.json so we are using 01/001/0078
                data_info = load_json(os.path.join(sg_fpath, 'AnotherMissOh01/001/0078', 'custom_data_info.json'))
        
            idx2node = data_info['ind_to_classes']             
            idx2relation = data_info['ind_to_predicates'] 
            sg_data = load_json(os.path.join(sg_file, 'custom_prediction.json'))
            for k, sg_item in sg_data.items():

                for idx, relation in enumerate(sg_item['rel_pairs']):
                    node1_idx, node2_idx = relation

                    node1 = idx2node[sg_item['bbox_labels'][node1_idx]]
                    node2 = idx2node[sg_item['bbox_labels'][node2_idx]]

                    node1_score = sg_item['bbox_scores'][node1_idx]
                    node2_score = sg_item['bbox_scores'][node2_idx]
                    relation = idx2relation[sg_item['rel_labels'][idx]]
                    relation_score = sg_item['rel_scores'][idx]

                    if (node1_score> args.bbox_threshold) and (node2_score> args.bbox_threshold) and (relation_score> args.rel_threshold):
                        phrases.append(node1 + ' '+ relation + ' ' + node2)

    else:
        ipdb.set_trace()

    return phrases

def demo_AnotherMissOh_sg(args, sg_fpath):
    sg_files = glob.glob(os.path.join(sg_fpath, '*.json'))
    sg_data_list = [load_json(sg_file) for sg_file in sg_files]
    phrases = []
    data_info = load_json(os.path.join(args.root_dir, 'demo', 'custom_data_info.json'))
    for sg_data in sg_data_list:
       
        idx2node = data_info['ind_to_classes']             
        idx2relation = data_info['ind_to_predicates'] 
        
        for k, sg_item in sg_data.items():

            for idx, relation in enumerate(sg_item['rel_pairs']):
                node1_idx, node2_idx = relation

                node1 = idx2node[sg_item['bbox_labels'][node1_idx]]
                node2 = idx2node[sg_item['bbox_labels'][node2_idx]]

                node1_score = sg_item['bbox_scores'][node1_idx]
                node2_score = sg_item['bbox_scores'][node2_idx]
                relation = idx2relation[sg_item['rel_labels'][idx]]
                relation_score = sg_item['rel_scores'][idx]

                if (node1_score> args.bbox_threshold) and (node2_score> args.bbox_threshold) and (relation_score> args.rel_threshold):
                    phrases.append(node1 + ' '+ relation + ' ' + node2)
    return phrases

def get_dataset_fn(args, dataset_type, tokenizer, split):
    if dataset_type == 'coqa':
        return get_coqa_dataset(args, tokenizer, split)
    return

def get_coqa_dataset(args, tokenizer, split):
    from datasets import load_dataset
    dataset = load_dataset("coqa")
    
    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    @dataclass
    class DataCollator_custom:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.nlp = spacy.load('en_core_web_lg')
        def __call__(self, pre_batch):
            Instruction = 'Given the story, answer the question.\n'
            texts_clean = []
            texts_aug = []
            for sample in pre_batch:
                question_clean = sample['questions'][0]
                question_aug = make_unanswerable(question_clean, ['zebra'])
                input_text_clean = Instruction+ 'story: ' + sample['story'] + '\nquestion: ' + question_clean + '\nanswer:'
                input_text_aug = Instruction+ 'story: ' + sample['story'] + '\nquestion: ' + question_aug + '\nanswer:'
                texts_clean.append(input_text_clean)
                texts_aug.append(input_text_aug)
            #stories = [Instruction+ 'story: ' + sample['story'] + '\nquestion: ' + sample['question'][0] + '\nanswer: ' for sample in pre_batch]

            data_clean = self.tokenizer(texts_clean, return_tensors="pt")
            data_augmented = self.tokenizer(texts_aug, return_tensors="pt")
            #------------------------
            return data_clean, data_augmented

    if split == 'validation':
        dataloader = DataLoader(dataset['validation'], batch_size=args.bsz, shuffle=False, num_workers=0, collate_fn=DataCollator_custom(tokenizer))
    return dataloader

def make_unanswerable(question, replacement_list):
    # Tokenize the question
    tokens = word_tokenize(question)
    
    # POS tagging
    tagged = pos_tag(tokens)
    
    # Separate nouns into proper nouns and common nouns
    proper_nouns = [word for word, pos in tagged if pos == "NNP"]
    common_nouns = [word for word, pos in tagged if pos == "NN"]
    
    # Prioritize replacing a proper noun. If none are found, consider replacing a common noun.
    if proper_nouns:
        noun_to_replace = choice(proper_nouns)
    elif common_nouns:
        noun_to_replace = choice(common_nouns)
    else:
        return question
    
    # Choose a replacement noun
    replacement = choice(replacement_list)
    
    # Replace the noun in the tokens list
    modified_tokens = [replacement if word == noun_to_replace else word for word in tokens]
    
    # Construct the modified question
    modified_question = ' '.join(modified_tokens)
    
    return modified_question