# Answerability Reasoning in VQA (ver2)

<!-- ABOUT THE PROJECT -->
## About The Project

Visual Question Answering (VQA) is the task of answering a question about an image. Evaluating if the posed question is relevant to the input image has gained some attention in the past because of its significance concerning the models' reliability when applied in real-life situations. However, previous approaches have yet to focus on reasoning about why a question is relevant/irrelevant to the given image. In this project, we develop a method that reasons the answerability of the question given an image. 

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
we are using DramaQA dataset from this year for answerability reasoning in Video Question Answering

Download scene graphs raw data from: https://nlp.stanford.edu/data/gqa/sceneGraphs.zip

Download questions raw data from: https://nlp.stanford.edu/data/gqa/questions1.2.zip

Put sceneGraph files: `train_sceneGraphs.json` and `val_sceneGraphs.json` into `gqa_data/`

Put questions json files: `train_balanced_questions.json` and `val_balanced_questions.json` into `gqa_data/`

After this step, the data file structure should look like this:

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


      train_sceneGraphs.json
      val_sceneGraphs.json
      train_balanced_questions.json
      val_balanced_questions.json
   object_list_extraction.py
   dataset_creation.py
   main.py
```