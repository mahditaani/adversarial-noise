

# Adversarial Noise Testing App

This application allows users to test adding adversarial noise to an image. It is built using **PyTorch** and has been tested with **Python 3.11.9**.

The app is built using **Streamlit** and allows users to upload an image and select the type of noise to add to the image. The app then displays the original image and the image with the added noise.

For further information, please refer to the following sources
- [Goodfellow et al. (2014)](https://arxiv.org/abs/1412.6572)
- [interpretable-ml-book](https://christophm.github.io/interpretable-ml-book/adversarial.html).
- [pytorch example](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

## Installation
Follow the steps below to set up the project:

1. Create a virtual environment:
```console
python -m venv .venv
```

1. Activate the virtual environment:
- On Windows:

    ```console
    .venv/bin/activate
    ```
- On MacOS/Linux:
    ```console
    source .venv/bin/activate
    ```

1. Install the required packages:
- Basic requirements:
    ```console
    pip install -r requirements.txt
    ```
- Development requirements:
    ```console
    pip install -r requirements.dev.txt
    ```

## Running the app
To run the app, execute the following command:
```console
streamlit run Home.py
```

## Running the tests
To run the tests, execute the following command:
```console
python -m pytest tests
```

## License
Don't know yet 😭

## TODO
- [ ] More thorough testing (ensuring data types are correct etc) 
- [ ] Make Attack class more generic to allow for different types of attacks (single pixel etc)
- [ ] Add Makefile
- [ ] Add CI/CD
- [ ] Generate documentation (Sphinx)
- [ ] Fix line length issues in code
- [ ] Check code coverage
- [ ] Add Dockerfile
- [ ] Add license 🤣