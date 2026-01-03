import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import torch
import datetime
import os


def load_model(model_id: str = "teknium/OpenHermes-2.5-Mistral-7B") -> tuple:
    """Load the LLM and its corresponding tokenizer."""
    tokenizer_path = "./mistral/tokenizer"
    if not os.path.isdir(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_path = "/tmp/mistral/model"
    if os.path.exists(model_path):
        model_llm = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model_llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
        )
        model_llm.save_pretrained(model_path)

    model_llm.config.pad_token_id = tokenizer.pad_token_id
    return model_llm, tokenizer


def get_prompt_template():
    return """<|im_start|>system
You are a Stockholm Pollen Expert. Help users with grass pollen levels.
- Use the context provided to generate your answer.
- Analyze if levels are safe for allergy sufferers.
- Do not show calculations or mention the context table.
<|im_end|>
### CONTEXT:
{context}
IMPORTANT: Today is {date_today}.
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


def get_llm_chain(model_llm, tokenizer):
    """Creates a modern LCEL chain (Runnable)."""
    text_gen_pipeline = transformers.pipeline(
        model=model_llm,
        tokenizer=tokenizer,
        task="text-generation",
        use_cache=True,
        do_sample=True,
        temperature=0.4,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    prompt = PromptTemplate(
        input_variables=["context", "question", "date_today"],
        template=get_prompt_template(),
    )

    # Modern LCEL Chain: Prompt | LLM | Parser
    return prompt | llm | StrOutputParser()


def generate_response_openai(
    user_query: str,
    feature_view,
    pollen_model,
    client,  # This is the OpenAI client or API key string
    verbose=False,
):
    """
    Generates a response using the modern LangChain ChatOpenAI implementation.
    """
    from functions.context_engineering import get_context_data

    # Use the existing context retrieval logic
    context = get_context_data(
        user_query,
        feature_view,
        pollen_model,
        client=client,
    )

    date_today = f'{datetime.date.today().strftime("%A")}, {datetime.date.today()}'

    if verbose:
        print(f"🗓️ Today's date: {date_today}")
        print(f"📖 {context}")

    # Initialize the modern 1.x compliant ChatOpenAI class
    # If 'client' passed from app is the wrapper, we extract the API key
    api_key = client.api_key if hasattr(client, "api_key") else client

    llm = ChatOpenAI(model="gpt-4-0125-preview", openai_api_key=api_key, temperature=0.2)

    # Get instructions and format them
    instructions_template = get_prompt_template().split("<|im_start|>user")[0]
    instructions_filled = instructions_template.format(context=context, date_today=date_today)

    # Invoke using the standard LCEL message format
    messages = [
        SystemMessage(content=instructions_filled),
        HumanMessage(content=user_query),
    ]

    response = llm.invoke(messages)

    return response.content.strip()


def generate_response(user_query, feature_view, model_pollen, model_llm, tokenizer, llm_chain, verbose=False):
    from functions.context_engineering import get_context_data

    context = get_context_data(user_query, feature_view, model_pollen, model_llm=model_llm, tokenizer=tokenizer)
    date_today = f'{datetime.date.today().strftime("%A")}, {datetime.date.today()}'

    # Use .invoke() for LCEL chains
    response = llm_chain.invoke(
        {
            "context": context,
            "date_today": date_today,
            "question": user_query,
        }
    )
    return response.split("<|im_start|>assistant")[-1].strip()
