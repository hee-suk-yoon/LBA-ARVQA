# Answerability Reasoning in Question Answering for Video (ver2)

<!-- ABOUT THE PROJECT -->
## About The Project

Visual Question Answering (VQA) is the task of answering a question about a visual input. Evaluating if the posed question is relevant to the input image (video) has gained some attention in the past because of its significance concerning the models' reliability when applied in real-life situations. However, previous approaches have yet to focus on reasoning about why a question is relevant/irrelevant to the given image (video). In this project, we develop a method that reasons the answerability of the question given an image (video). 

<!-- Usage -->
## Usage

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### 0. Dependencies

create a conda environment from `enviromentve_ver2.yaml` file.


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


  <p align="right">(<a href="#readme-top">back to top</a>)</p>

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
  2nd_year/
    AnotherMissOh_object_list_extraction.py
    AnotherMissOh_dataset_creation.py
    AnotherMissOh_preprocess_creation.py (optional)
    AnotherMissOh_main.py
```

### 2. Object List Extracting from Questions
Following commands should run without error:
```sh
python AnotherMissOh_object_list_extraction.py
```

### 3. Dataset augmentation to train the reasoning model
Following commands should run without error:
```sh
python AnotherMissOh_dataset_creation.py --data_name [train/val/test]
```

### 4. Preprocess the train/val/test dataset for fast experiment (optional)
When you run the main code, it contains the data preprocessing procedure. Since it takes quite few times, we provide the preprocess code to save the preprocessed data for further uses.  
Following commands should run without error:
```sh
python AnotherMissOh_preprocess_creation.py --save_name [SAVE_FILE_NAME] --data_split [train/val/test]
```

### 5. Training
For now, we only have single GPU training:
```sh
CUDA_VISIBLE_DEVICES=0 python AnotherMissOh_main.py --do_train --save_criterion loss
```

If you have preprocessed data, you need to add `--do_preprocess` and preprocess dataset path (i.e., `--preprocessed_train_data`, `--preprocessed_valid_data`)

### 6. Inference
For the inference only argument needs to be changed:
```sh
CUDA_VISIBLE_DEVICES=0 python AnotherMissOh_main.py --do_test
```
If you have preprocessed data, you need to add `--do_preprocess` and preprocess dataset path (i.e., `--preprocessed_test_data`)

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
