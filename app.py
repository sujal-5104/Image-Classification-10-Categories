
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Application configuration
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_image_classification_model.keras"
CONFIG_PATH = PROJECT_DIR / "models" / "app_config.json"

st.set_page_config(
    page_title="Image Classification AI",
    page_icon="🖼️",
    layout="centered"
)

# Load configuration
with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

CLASS_NAMES = config["class_names"]
CLASS_MAPPING = config["class_mapping"]
IMG_HEIGHT = config["image_height"]
IMG_WIDTH = config["image_width"]

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Application interface
st.title("🖼️ Image Classification AI")
st.subheader("10-Category Object Recognition")

st.write(
    "Upload an image to identify which of the "
    "10 object categories it belongs to."
)

st.info(
    "Educational machine learning project. "
    "Predictions are model estimates and may be incorrect."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Classify Image", type="primary"):

        with st.spinner("Analyzing image..."):

            resized_image = image.resize(
                (IMG_WIDTH, IMG_HEIGHT)
            )

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            probabilities = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            category_id = CLASS_NAMES[predicted_index]
            predicted_category = CLASS_MAPPING[category_id]

            confidence = float(
                probabilities[predicted_index] * 100
            )

        st.success("Classification completed!")

        st.metric(
            "Predicted Object",
            predicted_category
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # Display top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]

        st.subheader("Top 3 Predictions")

        for index in top_indices:

            category = CLASS_MAPPING[CLASS_NAMES[index]]
            score = float(probabilities[index] * 100)

            st.write(f"**{category}: {score:.2f}%**")
            st.progress(score / 100)

st.divider()

st.caption(
    "PRAICP-1010 | Image Classification with 10 Categories"
)
