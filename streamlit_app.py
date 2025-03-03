import streamlit as st
from openai import OpenAI
import os
from utils import extract_user_info
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core import SimpleDirectoryReader
from loguru import logger
from io import StringIO

# Show title and description.
st.title("📄 Data extraction using Gen AI")
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
if st.secrets.get("OPENAI_API_KEY") is not None:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


if not st.secrets["OPENAI_API_KEY"]:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document ('png', 'jpeg', 'gif', 'webp')",
        type=("png", "jpeg", "gif", "webp"),
    )

    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        with open(f"./{uploaded_file.name}", "wb") as file:
            file.write(bytes_data)
        image_documents = SimpleDirectoryReader(
            input_files=[f"./{uploaded_file.name}"]
        ).load_data()
        # # Generate an answer using the OpenAI API.
        try:
            response = extract_user_info(f"./{uploaded_file.name}")
        except Exception as e:
            logger.error(f"Error: {e}")
            response = "Sorry, an error occurred. Please try again."
        # # Stream the response to the app using `st.write_stream`.
        st.write(response)
        os.remove(f"./{uploaded_file.name}")
