import streamlit as st
from openai import OpenAI
import os
from utils import IdentityCard, prompt_template_str, gpt_4o
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core import SimpleDirectoryReader


# Show title and description.
st.title("📄 Data extraction using Gen AI")
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
if st.secrets.get("OPENAI_API_KEY") is not None:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_api_key = os.getenv("OPENAI_API_KEY")


@st.cache_resource(show_spinner="Model loading...")
def load_model():
    return MultiModalLLMCompletionProgram.from_defaults(
        output_cls=IdentityCard,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gpt_4o,
    )


model = load_model()

if not openai_api_key:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md", "jpg")
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
            response = model(image_documents=image_documents)
        except Exception as e:
            response = "Sorry, an error occurred. Please try again."
        # # Stream the response to the app using `st.write_stream`.
        st.write(response)
        os.remove(f"./{uploaded_file.name}")
