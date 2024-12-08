import json

import streamlit as st
from torchvision import models

from src.constants import IMAGENET_LABELS_JSON


# Caching the labels to make them available across pages
@st.cache_data
def load_labels():
    return json.load(open(IMAGENET_LABELS_JSON, "r"))


# Caching the model to avoid reloading
@st.cache_resource
def load_model(option: str):
    if option in SUPPORTED_MODELS:
        model_fn, weights = SUPPORTED_MODELS[option]
        model = model_fn(weights=weights)
        model.eval()
        return model
    else:
        raise ValueError(f"Model {option} not supported.")


SUPPORTED_MODELS = {
    "AlexNet": (models.alexnet, models.AlexNet_Weights.DEFAULT),
    "GoogLeNet": (models.googlenet, models.GoogLeNet_Weights.DEFAULT),
    "ResNet": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}
