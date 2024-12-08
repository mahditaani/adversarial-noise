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
st.write("This page allows you to perform adversarial attacks on images using pre-trained models using a range of epsilon and iterations.")
st.write("Upload an image and then carry out the attack which will generate adversarial noise with various different parameters. You can see how well the attack worked and how much the image was changed.")

# Sidebar for parameters
st.sidebar.header("Parameters")
model_name = st.sidebar.selectbox("Select Model", ["AlexNet", "GoogLeNet", "ResNet"])
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

epsilon_list = [0., 0.001, 0.005, 0.01]
iterations_list = [10, 20, 30]

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
            # Save uploaded file temporarily
            input_image.save(temp_file_path)
            st.header("Adversarial Image Attack Results")
            st.write(f"Running sweep for epsilon and iterations...")
            st.subheader(f"Target class: {target_class}")
            for epsilon in epsilon_list:
                for iterations in iterations_list:
                    if epsilon == 0. and iterations > 10:
                        continue
                    st.subheader(f"Epsilon: {epsilon}, Iterations: {iterations}")
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
                    res_col1, res_col2 = st.columns(2)
                    # highlight the box if the label is  the target class
                    

                    with res_col1:
                        # Display results
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        pipeline.visualize_attack(
                            input_img_tensor, output_img_tensor, ax=ax, show=False
                        )
                        res_col1.pyplot(fig)
                    with res_col2:
                        colour = "green" if result_adv_label == labels[str(target_class)] else "red"
                        res_col2.write(
                            f"**Original Image Label:** {result_original_label.split(',')[0]} - Confidence: {result_original_percentage:.2f}%"
                        )
                        res_col2.write(
                            f"**Adversarial Image Label:** :{colour}[{result_adv_label.split(',')[0]}] - Confidence: {result_adv_percentage:.2f}%"
                        )