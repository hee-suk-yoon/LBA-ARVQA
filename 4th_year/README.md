# Ambiguous Video QA Dataset Generation (4th_year)

<!-- ABOUT THE PROJECT -->
## About The Project

In this project, we focus on generating ambiguous video-grounded questions whose answers are present in the video but become clear only after a brief user clarification. We utilize Large Language Models (LLMs) like GPT-4 and Gemini to generate these questions based on video frames.

<!-- Usage -->
## Usage

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### 0. Dependencies

You will need to install the necessary Python packages. You can install them using pip:

```sh
pip install openai google-generativeai ray opencv-python numpy tqdm transformers datasets
```

### 1. Preprocess Video Frames
For fast inference and generation, we first convert videos into frames and save them as JSON files.
Please modify the `video_path_list` in `data_generation/save_frames.py` to point to your video folders.

```sh
python data_generation/save_frames.py
```
This will create a `preprocessed_frames` directory inside your video source directories.

### 2. Generate Dataset
We provide scripts to generate data using either OpenAI's GPT or Google's Gemini models.

#### Using ChatGPT
Modify `video_preprocessed_path_list` in `data_generation/data_generation_chatgpt.py` to point to your preprocessed frames.

```sh
python data_generation/data_generation_chatgpt.py --api_key [YOUR_OPENAI_API_KEY] --root_dir [OUTPUT_ROOT_DIR]
```

#### Using Gemini
Modify `video_preprocessed_path_list` in `data_generation/data_generation_gemini.py` to point to your preprocessed frames.

```sh
python data_generation/data_generation_gemini.py --gemini-api_key [YOUR_GEMINI_API_KEY] --root_dir [OUTPUT_ROOT_DIR]
```

### 3. Post-process Data
After generation, you can convert the generated data into a chat format suitable for training.
Modify `gen_data_path_list` and `video_root_path_dict` in `data_generation/preprocess.py`.

```sh
python data_generation/preprocess.py
```

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

<!-- CONTACT -->
## Contact

Eunseop Yoon - esyoon97@kaist.ac.kr

Hee Suk Yoon - hskyoon@kaist.ac.kr

Project Link: [https://github.com/tomyoon2/LBA-ARVQA](https://github.com/tomyoon2/LBA-ARVQA)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


