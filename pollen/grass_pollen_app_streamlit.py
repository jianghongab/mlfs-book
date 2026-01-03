import streamlit as st
import hopsworks
import joblib
import pandas as pd
import os
import ssl
from hopsworks import client

from dotenv import load_dotenv, find_dotenv
from xgboost import XGBRegressor
from openai import OpenAI
from functions.llm_chain import (
    load_model,
    get_llm_chain,
    generate_response,
    generate_response_openai,
)
import warnings

warnings.filterwarnings("ignore")


# 修复 Mac 上 Python 找不到证书的问题
if not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(ssl, "_create_unverified_context", None):
    ssl._create_default_https_context = ssl._create_unverified_context
# Set page configuration
st.set_page_config(page_title="Pollen AI Assistant", page_icon="🌿")
st.title("🌿 Stockholm Pollen AI Assistant 💬")


@st.cache_resource()
def connect_to_hopsworks():

    load_dotenv(override=True)
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")

    project = hopsworks.login(project=project_name, api_key_value=api_key)

    if project is None:
        raise ConnectionError("Login returned None without exception.")

    fs = project.get_feature_store()
    mr = project.get_model_registry()

    feature_view = None
    try:
        feature_view = fs.get_feature_view(
            name="pollen_weather_final",
            version=1,
        )
        if feature_view:
            feature_view.init_batch_scoring(1)
    except Exception as e:
        st.warning(f"Could not load feature view: {e}")

    model_pollen = None
    try:
        retrieved_model = mr.get_model(
            name="grass_pollen_model",
            version=4,
        )
        saved_model_dir = retrieved_model.download()
        model_pollen = joblib.load(saved_model_dir + "/pollen_model.pkl")
    except Exception as e:
        st.warning(f"Using local model. Hopsworks error: {e}")
        local_model_path = os.path.join(os.path.dirname(__file__), "pollen_model", "pollen_model.pkl")
        if os.path.exists(local_model_path):
            model_pollen = joblib.load(local_model_path)
        else:
            raise FileNotFoundError(f"Model not found at {local_model_path}")

    return feature_view, model_pollen


@st.cache_resource()
def retrieve_llm_chain():
    # Load the Local LLM and its corresponding tokenizer
    model_llm, tokenizer = load_model()

    # Create and configure the language model chain
    llm_chain = get_llm_chain(
        model_llm,
        tokenizer,
    )

    return model_llm, tokenizer, llm_chain


# --- Data Initialization ---

# Connect to Hopsworks and load resources
try:
    feature_view, model_pollen = connect_to_hopsworks()
except Exception as e:
    st.error(f"Failed to connect to Hopsworks: {e}")
    st.stop()

# Initialize session state for chat history and source selection
if "messages" not in st.session_state:
    st.session_state.messages = []
if "response_source" not in st.session_state:
    st.session_state.response_source = "OpenAI API"

# Sidebar configuration for model selection
st.sidebar.header("Configuration")
import platform
if platform.system() == "Windows":
    st.sidebar.warning("⚠️ Hermes LLM is not supported on Windows. Using OpenAI API only.")
    new_response_source = "OpenAI API"
else:
    new_response_source = st.sidebar.radio("Choose the response generation method:", ("Hermes LLM", "OpenAI API"), index=1)

# Clear chat history if the response source is changed
if new_response_source != st.session_state.response_source:
    st.session_state.messages = []
    st.session_state.response_source = new_response_source
    st.rerun()

# Handle API Key for OpenAI
openai_api_key = None
if new_response_source == "OpenAI API":
    # Try to load from .env first
    load_dotenv(override=True)
    env_api_key = os.getenv("OPENAI_API_KEY")
    if env_api_key and not env_api_key.startswith("<"):
        openai_api_key = env_api_key
        client = OpenAI(api_key=openai_api_key)
        st.sidebar.success("Using API key from .env file ✅")
    else:
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
        if openai_api_key:
            client = OpenAI(api_key=openai_api_key)
            st.sidebar.success("API key saved successfully ✅")
elif new_response_source == "Hermes LLM":
    try:
        model_llm, tokenizer, llm_chain = retrieve_llm_chain()
    except Exception as e:
        st.error(f"Failed to load Hermes LLM: {e}. Please use OpenAI API instead.")
        st.stop()

# --- Chat Interface ---

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_query := st.chat_input("How can I help you with pollen levels?"):
    # Display user message and add to history
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    st.write("⚙️ Generating Response...")

    if new_response_source == "OpenAI API" and openai_api_key:
        response = generate_response_openai(
            user_query,
            feature_view,
            model_pollen,
            client,
            verbose=False,
        )
    elif new_response_source == "Hermes LLM":
        # Use Local LLM to generate response
        response = generate_response(
            user_query,
            feature_view,
            model_pollen,
            model_llm,
            tokenizer,
            llm_chain,
            verbose=False,
        )
    else:
        response = "Please provide the necessary credentials in the sidebar."

    # Display assistant response and update history
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.info("This assistant provides predictions and advice for Stockholm grass pollen levels based on historical weather and pollen data.")
