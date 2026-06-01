from pathlib import Path

import streamlit as st

from src.predict import DEFAULT_MODEL_PATH, predict_character

st.set_page_config(page_title="Handwritten Character Recognition")
st.title("Handwritten Character Recognition")
st.write("Upload one handwritten digit or A-Z letter.")

if not Path(DEFAULT_MODEL_PATH).exists():
    st.warning("Train the model first with: python src/train.py")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", width=200)
    try:
        character, confidence = predict_character(uploaded_file)
        st.success(f"Prediction: {character}")
        st.write(f"Confidence: {confidence:.2%}")
    except FileNotFoundError as error:
        st.error(str(error))
