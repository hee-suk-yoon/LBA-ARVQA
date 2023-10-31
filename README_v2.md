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

### 1. Downlaod Dataset
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
                AnotherMissOh02/
                    .
                    .
                    .

   AnotherMissOh_object_list_extraction.py
   AnotherMissOh_dataset_creation.py
   AnotherMissOh_preprocess_creation.py (optional)
   AnotherMissOh_main.py
```