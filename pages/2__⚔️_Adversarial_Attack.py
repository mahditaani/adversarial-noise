import io
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from app import load_labels, load_model
from src.attack_pipeline import AdversarialAttackPipeline

# Load model and labels
labels = load_labels()

# App title
st.title("Adversarial Image Attack Tool")
st.write("This page allows you to perform adversarial attacks on images using pre-trained models.")
st.write("Upload an image and then adjust the parameters to generate an adversarial image.")

# Sidebar for parameters
st.sidebar.header("Parameters")
model_name = st.sidebar.selectbox("Select Model", ["AlexNet", "GoogLeNet", "ResNet"])
epsilon = st.sidebar.slider(
    "Epsilon (Perturbation Strength)",
    0.001,
    0.01,
    0.001,
    step=0.001,
    format="%.3f",
    help="Controls the intensity of the adversarial noise added to the image. Larger epsilon introduces more noticeable changes.",
)
iterations = st.sidebar.slider(
    "Iterations",
    1,
    100,
    30,
    step=1,
    help="Number of optimization steps used to create the adversarial image.",
)
target_class = st.sidebar.number_input(
    "Target Class (Imagenet Index)",
    value=368,
    step=1,
    min_value=0,
    max_value=999,
    help="Specify a desired misclassification target (e.g., 'tiger'). The attack modifies the image to maximize the model's confidence in this target class.",
)
dynamic_epsilon = st.sidebar.checkbox(
    "Dynamic Epsilon",
    value=False,
    help="Adjust the perturbation strength dynamically (per pixel) during the attack process.",
)

model = load_model(model_name)
# File uploader
uploaded_file = st.file_uploader(
    "Upload an Image (JPEG/PNG)", type=["jpeg", "jpg", "png"]
)

if uploaded_file:
    # Display the uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")
    _, mid_col, _ = st.columns(3)
    with mid_col:
        mid_col.image(input_image, caption="Uploaded Image")

    # Button to perform attack
    if st.button("Perform Attack"):
        with TemporaryDirectory() as temp_dir:
            # Save uploaded file temporarily
            temp_file_path = f"{temp_dir}/temp_image.png"
            input_image.save(temp_file_path)

            # Initialize the attack pipeline
            pipeline = AdversarialAttackPipeline(
                model=model, labels=labels, epsilon=epsilon, iterations=iterations
            )

            # Perform the attack
            with st.spinner("Generating adversarial image..."):
                input_img_tensor, output_img_tensor = pipeline.perform_attack(
                    temp_file_path,
                    target_class=target_class,
                    dynamic_epsilon=dynamic_epsilon,
                )

            input_img = pipeline.image_processor.inverse_process(input_img_tensor)
            output_img = pipeline.image_processor.inverse_process(output_img_tensor)

            # Evaluate labels
            result_original_label, result_original_percentage = pipeline.evaluator.evaluate(
                input_img_tensor
            )
            result_adv_label, result_adv_percentage = pipeline.evaluator.evaluate(
                output_img_tensor
            )
            st.write(
                f"**Original Image Label:** {result_original_label.split(',')[0]} - Confidence: {result_original_percentage:.2f}%"
            )
            st.write(
                f"**Adversarial Image Label:** {result_adv_label.split(',')[0]} - Confidence: {result_adv_percentage:.2f}%"
            )

            # Display results
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            pipeline.visualize_attack(
                input_img_tensor, output_img_tensor, ax=ax, show=False
            )
            st.pyplot(fig)

            # Allow downloading of images
            col1, col2 = st.columns(2)

            with col1:
                input_buffer = io.BytesIO()
                input_img.save(input_buffer, format="PNG")
                input_buffer.seek(0)

                col1.download_button(
                    label="Download Original Image",
                    data=input_buffer,
                    file_name="original_image.png",
                    mime="image/png",
                )
            with col2:
                output_buffer = io.BytesIO()
                output_img.save(output_buffer, format="PNG")
                output_buffer.seek(0)

                col2.download_button(
                    label="Download Adversarial Image",
                    data=output_buffer,
                    file_name="adversarial_image.png",
                    mime="image/png",
                )
