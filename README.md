

# Adversarial Noise Testing App

This application allows users to test adding adversarial noise to an image. It is built using **PyTorch** and has been tested with **Python 3.11.9**.

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
streamlit run main.py
```

## Running the tests
To run the tests, execute the following command:
```console
python -m pytest tests
```

## License
Don't know yet 😭

## TODO
- [ ] Add CI/CD
- [ ] Generate documentation (Sphinx)
- [ ] Fix line length issues in code
- [ ] Check code coverage
- [ ] Add Dockerfile
- [ ] Add license 🤣