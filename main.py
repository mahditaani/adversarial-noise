import streamlit as st

st.set_page_config(page_title="Adversarial Image Attack Tool", layout="wide")

st.markdown(
    """
    # Adversarial Image Attack Tool

    Welcome to the **Adversarial Image Attack Tool**! This tool demonstrates how adversarial attacks can subtly modify an image to deceive deep learning models. It's both an educational resource and a way to visualize the vulnerabilities of machine learning models like neural networks.

    ## What are Adversarial Attacks?

    Adversarial attacks are intentional manipulations of input data designed to fool machine learning models. In the context of image classification, adversarial attacks make small, often imperceptible changes to an image that cause a model to misclassify it. These attacks highlight the challenges in deploying robust and secure AI systems.

    ### How Does It Work?

    This tool performs adversarial attacks on images using the following steps:

    1. **Model Selection**: We use a pre-trained [AlexNet](https://pytorch.org/vision/stable/models.html#alexnet) model, a popular convolutional neural network, trained on the ImageNet dataset.

    2. **Adversarial Attack Pipeline**:
    - The original image is fed into the model.
    - Small perturbations are applied to the image using a method called **gradient-based optimization**, aimed at causing the model to misclassify the image.
    - Parameters such as `epsilon` (perturbation strength) and the number of `iterations` determine the severity of the attack.

    3. **Visualization and Results**:
    - The tool visualizes the original and adversarial images side by side.
    - It shows the classification labels and confidence levels for both the original and modified images.

    4. **Downloadable Outputs**:
    - You can download both the original and adversarial images for further exploration.

    ## Tool Parameters

    - **Epsilon (Perturbation Strength)**:
    This parameter controls the intensity of the adversarial noise added to the image. A larger epsilon introduces more noticeable changes, potentially making the adversarial image less natural.

    - **Iterations**:
    The number of optimization steps used to create the adversarial image. More iterations generally result in more effective attacks but take longer to compute.

    - **Target Class**:
    You can specify a desired misclassification target (e.g., "tiger"). The attack then modifies the image to maximize the model's confidence in this target class.

    - **Dynamic Epsilon**:
    An optional setting that adjusts the perturbation strength dynamically during the attack process.

    ## How to Use the Tool

    1. **Upload an Image**:
    Upload any image in JPEG or PNG format using the file uploader.

    2. **Set Parameters**:
    Use the sidebar to adjust parameters like epsilon, iterations, and target class.

    3. **Perform Attack**:
    Click the **Perform Attack** button to generate an adversarial image.

    4. **View Results**:
    The tool displays:
    - The original and adversarial images side by side.
    - The classification labels and confidence levels for each image.

    5. **Download Results**:
    Save the original and adversarial images to your device using the download buttons.

    ## Why is this Important?

    Adversarial attacks are a critical area of research in AI security. They reveal potential weaknesses in models, helping researchers develop more robust systems. By using this tool, you can gain hands-on experience with adversarial examples and learn more about the inner workings of AI models.

    ---

    We hope you find this tool insightful and educational. If you have any questions or feedback, feel free to share!

    """
)
