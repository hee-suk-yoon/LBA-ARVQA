# Answerability Reasoning in Question Answering for Video (ver2)

<!-- ABOUT THE PROJECT -->
## About The Project

Visual Question Answering (VQA) is the task of answering a question about a visual input. Evaluating if the posed question is relevant to the input image (video) has gained some attention in the past because of its significance concerning the models' reliability when applied in real-life situations. However, previous approaches have yet to focus on reasoning about why a question is relevant/irrelevant to the given image (video). In this project, we develop a method that reasons the answerability of the question given an image (video). 

<!-- Usage -->
## Usage

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### 0. Dependencies

create a conda environment from `3rd_year_video_understadning/environment.yaml` file.


  ```sh
  conda env create --name ENV_NAME --file environment.yaml
  ```

  <p align="right">(<a href="#readme-top">back to top</a>)</p>

### 1. Download Dataset
The dataset is currently being generated. TBU

### N. Output Example
Currently the output is natural language form as shown in the below example.
```sh
{
        "vid": "XWLh6BBJY2c",
        "question": "How does the boy with the blueish hat perform different basketball maneuvers compared to others in the video?",
        "gt_answer": "The question cannot be answered as there is no boy with a blueish hat featured in the video, only a boy with a white hat is shown playing basketball.",
        "predidction": "In the video, the boy with the blueish hat is seen performing a dribbling maneuver, which involves bouncing the basketball on the ground while keeping it in motion. This is different from the other players who are seen standing around the basketball court, possibly waiting for their turn to play or observing the game. The boy with the blueish hat demonstrates his skill and agility by engaging in a specific basketball action, while the other players are more focused on the overall gameplay and strategy."
    }
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
