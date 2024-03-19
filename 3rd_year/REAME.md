# Answerability Reasoning in Question Answering for Video (3rd_year)

<!-- ABOUT THE PROJECT -->
## About The Project

Visual Question Answering (VQA) is the task of answering a question about a visual input. Evaluating if the posed question is relevant to the input image (video) has gained some attention in the past because of its significance concerning the models' reliability when applied in real-life situations. However, previous approaches have yet to focus on reasoning about why a question is relevant/irrelevant to the given image (video). In this project, we develop a method that reasons the answerability of the question given an image (video). 

<!-- Usage -->
## Usage

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### 0. Dependencies

create a conda environment from `enviromentve_ver2.yaml` file. (same as 2nd year)


  ```sh
  conda env create --name ENV_NAME --file environment_ver2.yaml
  ```


After activate the conda environment, download spaCy for english
  ```sh
  python -m spacy download en_core_web_lg
  ```

Please use the NLTK Downloader to obtain the resource:
 ```sh
  python 
  import nltk
  nltk.download('punkt')
  ```

### 1. Download Dataset
We are using DramaQA dataset from this year for answerability reasoning in Video Question Answering

Download questions raw data from: https://dramaqa.snu.ac.kr/ 

Generate scene graph following: https://github.com/youngyoung1021/cross-graph-attention 



For the easier running, we recommend the data file structure looks like this:

```sh
LBA-ARVQA
  dataset/
    DramaQA/
      AnotherMissOh_Visual_Faster_RCNN_detected.json
      AnotherMissOhQA_train_set.json
      AnotherMissOhQA_val_set.json
      AnotherMissOhQA_test_set.json
    AnotherMissOh/
      scene_graph/
        AnotherMissOh01/
          001/
            0078/
              custom_data_info.json
              custom_prediction.json
              .
              . 
        AnotherMissOh02/
            .
            .
            .
  3rd_year/
      demo/
          demo_sg/
            extracted_sg.json
            .
            .
        demo_question.json
        custom_data_info.json
    object_list_extraction.py
    main.py
    main_demo.py
    utils.py
    utils_sys.py
```

### 2. Object List Extracting from Questions
Following commands should run without error. It will save `AnotherMissOhQA_object_list.pkl` in `3rd_year/saves`: 
```sh
python object_list_extraction.py
```

### 3-1. Run the demo (main_demo.py)
The input for the demo needs to be prepared in advance (scene graph for the video and question as saved in 'demo' folder). The scene graph can be multiple json files but it needs to be saved in `demo/demo_sg` If you run this file, the output file will be saved in `demo/LBA_2024/output_KAIST.json`

```sh
python main_demo.py --root_dir {dir_of_3rd_year_foler} --model_ckpt {saved ckpt results from 2nd year} --classifier_ckpt {saved ckpt results from 2nd year}
```

### 3-2. Run main.py (main_demo.py)
main.py is specifically working for the input question is AnotherMissOh train or test dataset and corresponding scene graphs. If you run this `main.py` code, it will generate the `LBA_2024/output_KAIST.json` (it can be sepcified by `args.output_dir` and `args.output_fname`) and save the answerability prediction results for the whole dataset. Furthermore, when you set `args.generate_unanswerable_que`,  it will generate unawerable question and save the prediction on generated unanswerable question as well. (Default: `False`)

```sh
python main.py --root_dir {dir_of_3rd_year_foler} --model_ckpt {saved ckpt results from 2nd year} --classifier_ckpt {saved ckpt results from 2nd year}
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