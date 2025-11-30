import os
import json
import sys
from glob import glob
from tqdm import tqdm


gen_data_path_list = [
    "GENERATED_DATA_FOLDER_PATHS"
]

video_root_path_dict = {
    "DATA_KEY": "RAW_VIDEO_PATH_FOR_THE_DATA"
}

def preprocess_data_folder(file_path):
    raw_data = []
    file_list = glob(os.path.join(file_path, "*.json"))
    data_name = file_path.split("/")[-1]
    for file_name in tqdm(file_list, desc=f"Processing {file_path}"):
        sample = file_name.split("/")[-1]
        with open(file_name, 'r') as f:
            data = json.load(f)
            data['dataset'] = data_name
            data['video_idx'] = sample.replace('.json', '')
            raw_data.append(data)
    return raw_data

def processing_chat_format(data, ):
    video_home_path = video_root_path_dict[data['dataset']]
    video_path = glob(os.path.join(video_home_path, f"{data['video_idx']}.*"))
    if len(video_path) == 0:
        print(f"Video file not found for {data['video_idx']} in dataset {data['dataset']}")
        import ipdb; ipdb.set_trace()
    message = [
        {"role": "user", "content": f"<video>{data['ambiguous_question']}"},
        {"role": "assistant", "content": data['clarifying_question']},
        {"role": "user", "content": data['clarifying_answer_user']},
        {"role": "assistant", "content": f"Based on the given additional information, {data['final_answer']}"}
    ]   
    processed_data = {
        "videos": video_path,
        "messages": message
    }
    return processed_data

def main():
    data = []
    for path in gen_data_path_list:
        data += preprocess_data_folder(path)
    
    processed_data = []
    save_data_path = "SAVE_PATH_FOR_PROCESSED_DATA.json"
    for idx, data_item in enumerate(data):
        data_item['id'] = idx
        processed_item = processing_chat_format(data_item)
        processed_data.append(processed_item)

    with open(save_data_path, 'w') as f:
        json.dump(processed_data, f, indent=4)


if __name__ == "__main__":
    main()

