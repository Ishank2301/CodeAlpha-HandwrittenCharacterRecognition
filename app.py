import streamlit as st

from src.predict import predict

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered",
)

st.title("Handwritten Character Recognition")

st.markdown("""
    Upload a handwritten digit or alphabet image and select
    the model you want to use for prediction.
    """)

with st.sidebar:

    st.header("Model Selection")

    model_type = st.radio("Choose Model", ["MNIST", "EMNIST"])

    st.divider()

    st.markdown(f"""
        **Selected Model**

        {model_type}
        """)

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    try:

        with st.spinner("Generating prediction..."):

            prediction, confidence = predict(uploaded_file, model_type)

        with col2:

            st.subheader("Prediction")

            st.metric(label="Character", value=prediction)

            st.metric(label="Confidence", value=f"{confidence:.2%}")

    except Exception as error:

        st.error(f"{error}")
