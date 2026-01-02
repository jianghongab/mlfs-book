import streamlit as st
import hopsworks
import joblib
import pandas as pd
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

# Set page configuration
st.set_page_config(page_title="Pollen AI Assistant", page_icon="🌿")
st.title("🌿 Stockholm Pollen AI Assistant 💬")


@st.cache_resource()
def connect_to_hopsworks():
    # Initialize Hopsworks connection
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Retrieve the 'pollen_weather_final' feature view created in the notebook
    feature_view = fs.get_feature_view(
        name="pollen_weather_final",
        version=1,
    )

    # Initialize batch scoring for inference
    feature_view.init_batch_scoring(1)

    # Retrieve the 'grass_pollen_model' from the model registry
    retrieved_model = mr.get_model(
        name="grass_pollen_model",
        version=4,  # Matches the version saved in your notebook
    )

    # Download model artifacts
    saved_model_dir = retrieved_model.download()

    # Load the XGBoost regressor model using joblib
    # In your notebook, the model was saved as a .pkl file
    model_pollen = joblib.load(saved_model_dir + "/pollen_model.pkl")

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
new_response_source = st.sidebar.radio("Choose the response generation method:", ("Hermes LLM", "OpenAI API"), index=1)

# Clear chat history if the response source is changed
if new_response_source != st.session_state.response_source:
    st.session_state.messages = []
    st.session_state.response_source = new_response_source
    st.rerun()

# Handle API Key for OpenAI
openai_api_key = None
if new_response_source == "OpenAI API":
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        st.sidebar.success("API key saved successfully ✅")
elif new_response_source == "Hermes LLM":
    model_llm, tokenizer, llm_chain = retrieve_llm_chain()

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
        # Use OpenAI to generate response based on pollen model predictions
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
