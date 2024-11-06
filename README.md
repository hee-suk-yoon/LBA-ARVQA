# Answerability Reasoning in Question Answering for Video for 3rd year

<!-- ABOUT THE PROJECT -->
## About The Project

Visual Question Answering (VQA) is the task of answering a question about a visual input. Evaluating if the posed question is relevant to the input image (video) has gained some attention in the past because of its significance concerning the models' reliability when applied in real-life situations. However, previous approaches have yet to focus on reasoning about why a question is relevant/irrelevant to the given image (video). In this project, we develop a method that reasons the answerability of the question given an image (video). 

<!-- Usage -->
## Usage

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### 0. Dependencies

create a conda environment from `environment.yaml` file.


  ```sh
  cd 3rd_year_video_understanding
  conda env create --name ENV_NAME --file environment.yaml
  ```

Download a provided model. The path of this model would be `[SAVED_MODEL_PATH]` in the followed instruction.


### 1. Prepare Dataset
We can use any type of Video QA dataset in following format. Currently the example dataset is in the `sample` folder for the test. `demo_vids` folder is for the sample-wise inference.
```sh
LBA-ARVQA
  3rd_year_video_understanding/
    dataset/
      videos/
      frames/ (optional)
      (sample/ (for_test))
      data/
        qa_data.json

```
For the question data, the `qa_data.json` should look like 

```sh
[
    {
        "vid": "HPjONYEKwnY.mp4",
        "question": "How does the adult adjust the settings of the phototherapy machine in the video?",
        "answer": "The question is unanswerable because the video does not feature a phototherapy machine."
    },
    .
    .
    .
]
```

or
```sh
[
    {
        "vid": "HPjONYEKwnY",
        "question": "How does the adult adjust the settings of the phototherapy machine in the video?",
        "answer": "The question is unanswerable because the video does not feature a phototherapy machine."
    },
    .
    .
    .
]
```

When we input the video frames (`dataset/sample/data/sample_img.json`), the dataset should be the folder name of each video frames or when we input the video file itself (`dataset/sample/data/sample_vid.json`), the vid in the dataset should be the file name of the video.

### 2. Preprocess video into the frames (optional)
For the fast inference, you can convert the video into frames in advance. 
Please modify the `video_folder` into the folder of videos, and `output_folder` into the output folder for the frames.
```sh
python dataset/convert_vid_to_frames.py 
```

### 3. Inference 
When you have the dataset that you want to evaluate, run the following command
* Inference for demo and input data format: video 
```sh
python inference/run_inference_sample.py --model-path [SAVED_MODEL_PATH] --input_video
```

* Inference for demo and input data format: images (video converted into frames) 
```sh
python inference/run_inference_sample.py --model-path [SAVED_MODEL_PATH] --input_image_frames
```

* Inference for evaluating or large scale and input data format: video
```sh
python inference/run_inference_dataset.py --model-path [SAVED_MODEL_PATH] --input_video --save_name [SAVE_RESULT_FILE_NAME] --home_path [ABS_PATH_FOR_3rd_year_video_understanding] --data_name qa_data.json --data_path dataset 
```

* Inference for evaluating or large scale and input data format: images (video converted into frames)
```sh
python inference/run_inference_dataset.py --model-path [SAVED_MODEL_PATH] --input_image_frames --save_name [SAVE_RESULT_FILE_NAME] --home_path [ABS_PATH_FOR_3rd_year_video_understanding] --data_name qa_data.json --data_path dataset
```

For the large scale inference, you can use the sample dataset uploaded in this repository without setting `data_name` and `data_path`
<p align="right">(<a href="#readme-top">back to top</a>)</p>


### 4. Extracting the unanswerable entity (optional)
If you want to extract the specific entity of the reason why the question is unanswerable, run the following command. It will save the processed json file in the `result` folder
```sh
python inference/process_unanswerable_answer.py --pred_path [PREDICTION_FILE_FROM_STEP3] --home_path [ABS_PATH_FOR_3rd_year_video_understanding] --save_name [PROCESSED_FILE_NAME_FOR_SAVE] --api_key [OPENAI_KEY]
```
When you don't want to use GPT API, you can use llama-3-8b-instruct instead.
```sh
python inference/process_unanswerable_answer.py --pred_path [PREDICTION_FILE_FROM_STEP3] --home_path [ABS_PATH_FOR_3rd_year_video_understanding] --save_name [PROCESSED_FILE_NAME_FOR_SAVE] --llama3_path [MODEL_PATH_OF_LLAMA3] --run_model llama3
```

### 5. Run Inference for DramaQA samples (optional)
* Inference for DramaQA samples (image frames are placed in `sample/DramaQA`, sample json file is `dataset/sample/data/sample_dramaqa.json`)
```sh
python inference/run_inference_dataset.py --model-path [SAVED_MODEL_PATH] --input_image_frames --save_name [SAVE_RESULT_FILE_NAME] --home_path [ABS_PATH_FOR_3rd_year_video_understanding] --data_name sample_dramaqa.json --data_path dataset --do_dramaQA
```


You can cehck the example of the processed results is in `result/llama3_entity_extraction.json`
<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



LICENSE
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Hee Suk Yoon - hskyoon@kaist.ac.kr

Eunseop Yoon - esyoon97@kaist.ac.kr

Project Link: [https://github.com/tomyoon2/LBA-ARVQA](https://github.com/tomyoon2/LBA-ARVQA)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
