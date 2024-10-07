import os
import argparse
from tqdm import tqdm
import argparse
import torch
from typing import Sequence

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, "transformers-4.41-release", 'src')))


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, KeywordsStoppingCriteriaForBatch

from transformers import TextStreamer
import os
import shutil
import json
from glob import glob
from infer_utils import get_chunk, load_json, save_json, load_frames, load_jsonl, save_jsonl, select_n_frames_from_video, load_video_into_frames

from torch.utils.data import DataLoader # added by esyoon 2024-08-30-02:40:45
import torch.nn.functional as F # added by esyoon 2024-08-30-13:46:53


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/data/kakao/workspace/models/vlm_rlaif_video_llava_7b')
    parser.add_argument("--model-base", type=str, default=None)

    # Define the command-line arguments
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--num_frames', default=50, type=int)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--rlhf_ckpt", action="store_true", default=False, help="Whether it is form RLHF checkpoint")
    parser.add_argument("--resume", action="store_true", default=False, help="Whether to resume inference")

    parser.add_argument("--chunks", type=int)
    parser.add_argument("--chunk_idx", type=int)

    parser.add_argument("--run_model_name", type=str, default="vlm_rlaif", help="Model name for the run vlm_rlaif / video_llava / llama_vid")
    parser.add_argument("--input_video", action="store_true", help="Whether to use video as input")
    parser.add_argument("--input_image_frames", action="store_true", help="Whether to use image frames as input (already converted)")

    parser.add_argument("--save_name", type=str, default="pred_output.json", help="Name of the output file")
    parser.add_argument("--home_path", type=str, required=True, help="Path of LBA-ARVQA 3rd_year_video_understanding")
    parser.add_argument("--data_path", type=str, default="dataset/sample", help="Data path")
    parser.add_argument("--data_name", type=str, default="sample_img.json", help="Data file name saved under data_path")

    parser.add_argument("--batch_size", type=int, default=1)


    return parser.parse_args()

def _load_model(args):
    model_name = get_model_name_from_path(args.model_path)
    if args.run_model_name == "vlm_rlaif":
        model_name = "vlm_rlaif_video_llava_7b"
    if args.rlhf_ckpt:
        model_name = args.model_path # FIXME naive solution
        if not os.path.exists(os.path.join(args.model_path, "config.json")):
            shutil.copy(os.path.join(args.model_base, "config.json"), os.path.join(args.model_path, "config.json")) # Copy SFT model's config -> to RLHF folder
    tokenizer, model, image_processor, context_len = load_pretrained_model(args, args.model_path, args.model_base, 
                                                                           model_name, args.load_8bit, args.load_4bit, 
                                                                           device=args.device)
    
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        # conv_mode = "llava_v0"
        conv_mode = "llava_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    return model, tokenizer, image_processor, context_len, args.conv_mode


class UvqaEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data, frame_path):
        self.data = data
        self.frame_path = frame_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'vid': self.data[idx]['vid'], 
                'vid_path': os.path.join(self.frame_path, self.data[idx]['vid']),
                'question': self.data[idx]['question'],
                'answer': self.data[idx]['answer']
                }

    
        return sample

def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )  # noqa
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence

def get_conversation_prompt(args, model, question, images):
    # Get conversation
    conv = conv_templates[args.conv_mode].copy()
    if images is not None:
        # first message
        if model.__class__.__name__ == "VideoLlavaForConditionalGeneration":
            question = DEFAULT_VIDEO_TOKEN + question
        else:
            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], question)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], question)
        # conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def collate_fn(args, model, tokenizer, image_processor, batch):

    def _left_pad_helper(instances: Sequence[dict], key: str, pad_token: int):
        # TODO(lxuechen): Potentially replace with `transformers.PretrainedTokenizerBase.prepare_for_model`.
        # `instances` is a list of dicts, each dict has key whose value is a list of tensors, possibly of unequal length.
        # input_ids = [instance[key] for instance in instances]  # Flatten.
        input_ids = instances[key]
        try:
            input_ids = pad_sequence_from_left(
                input_ids,
                batch_first=True,
                padding_value=pad_token,
            )
        except:
            raise ValueError(f"Error padding {key} for {input_ids}")
        return input_ids
    
    sample = {}
    conv = conv_templates[args.conv_mode].copy()

    if args.input_image_frames:
        batch_images = [load_frames(glob(item['vid_path'] + '/*'), args.num_frames) for item in batch]
    elif args.input_video:
        batch_images = [load_video_into_frames(item['vid_path'], num_frames=args.num_frames, return_tensor=False) for item in batch]
    image_tensor = [process_images(images, image_processor, args) for images in batch_images]
    image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)
    prompt = [get_conversation_prompt(args, model, item['question'], images) for item, images in zip(batch, batch_images)]

    input_ids = [tokenizer_image_token(item, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for item in prompt]
    sample['input_ids'] = input_ids
    input_ids = _left_pad_helper(sample, 'input_ids', tokenizer.pad_token_id).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteriaForBatch(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    sample['input_ids'] = input_ids
    sample['stopping_criteria'] = stopping_criteria
    sample['streamer'] = streamer
    sample['images'] = image_tensor

    sample['vid'] = [item['vid'] for item in batch]
    sample['question'] = [item['question'] for item in batch]
    sample['answer'] = [item['answer'] for item in batch]
    return sample

def get_prompt(args, model, question, images=None):
    # Get conversation
    conv = conv_templates[args.conv_mode].copy()
    if images is not None:
        # first message
        if model.__class__.__name__ == "VideoLlavaForConditionalGeneration":
            question = DEFAULT_VIDEO_TOKEN + question
        else:
            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], question)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], question)
        # conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    if args.model_base == 'None':
        args.model_base = None

    LOG_INTERVAL = 50
    # Initialize the model
    if args.run_model_name == "vlm_rlaif":
        model, tokenizer, image_processor, context_len, args.conv_mode = _load_model(args)
    elif args.run_model_name == "video_llava":
        model = VideoLlavaForConditionalGeneration.from_pretrained(args.model_path)
        processor = VideoLlavaProcessor.from_pretrained(args.model_path)
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
        args.conv_mode = "llava_v1"

        model.cuda()
    

    data_root_dir = os.path.join(args.home_path, args.data_path)
    data_fname = os.path.join(data_root_dir, "data", args.data_name)
    # data_fname = os.path.join(data_root_dir, "data", 'sample_img.json')
    # data_fname = os.path.join(data_root_dir, "data", 'sample_vid.json')
    frame_path = os.path.join(data_root_dir, "frames") # when use the input video frames
    video_path = os.path.join(data_root_dir, "videos") # when use the input video 
    args.save_path = os.path.join(args.home_path, "result")
    save_fname = os.path.join(args.save_path, args.save_name + '.json')
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    output_save = []
    data = load_json(data_fname)
    if args.input_image_frames:
        dataset = UvqaEvalDataset(data, frame_path)
    elif args.input_video:
        dataset = UvqaEvalDataset(data, video_path)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(args, model, tokenizer, image_processor, x))
    for idx, batch in enumerate(tqdm(dataloader)):
        with torch.inference_mode():
            if args.run_model_name == "vlm_rlaif":
                output_ids = model.generate(
                    batch['input_ids'],
                    images=batch['images'],
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[batch['stopping_criteria']])
        outputs = tokenizer.batch_decode(output_ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        outputs = [output.strip() for output in outputs]
        current_save_list = []
        for i in range(len(outputs)):
            current_save_list.append({
                'vid': batch['vid'][i],
                'question': batch['question'][i],
                'gt_answer': batch['answer'][i],
                'pred': outputs[i],
            })
        output_save.extend(current_save_list)
        save_json(output_save, save_fname)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)