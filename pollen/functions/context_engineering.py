import xml.etree.ElementTree as ET
import re
import inspect
from typing import get_type_hints, Any, Dict, List
import json
import datetime
import torch
import sys
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from functions.pollen_data_retrieval import (
    get_historical_pollen_for_date,
    get_historical_pollen_in_date_range,
    get_future_pollen_for_date,
    get_future_pollen_in_date_range,
)


def serialize_function_to_json(func: Any) -> str:
    signature = inspect.signature(func)
    function_info = {
        "name": func.__name__,
        "description": func.__doc__.strip() if func.__doc__ else "Pollen retrieval.",
        "parameters": {"type": "object", "properties": {}},
        "returns": "DataFrame",
    }
    for name in signature.parameters:
        if name not in ["feature_view", "weather_fg", "model"]:
            function_info["parameters"]["properties"][name] = {"type": "string"}
    return json.dumps(function_info, indent=2)


def extract_function_calls(completion: str) -> List[Dict[str, Any]]:
    pattern = r"(<onefunctioncall>(.*?)</onefunctioncall>)"
    match = re.search(pattern, completion, re.DOTALL)
    if not match:
        return None
    try:
        root = ET.fromstring(match.group(1))
        return [json.loads(fn.text) for fn in root.findall("functioncall")]
    except:
        return None


def get_function_calling_prompt(user_query: str) -> str:
    functions_desc = f"{serialize_function_to_json(get_historical_pollen_for_date)}\n{serialize_function_to_json(get_future_pollen_in_date_range)}"
    return f"""<|im_start|>system
You are a Stockholm Pollen Expert. Use functions to find grass pollen levels.
### FUNCTIONS:
{functions_desc}
Respond STRICTLY with XML format: <onefunctioncall><functioncall> {{...}} </functioncall></onefunctioncall>
Today is {datetime.date.today().strftime("%A")}, {datetime.date.today()}.
<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant"""


def invoke_function(function, feature_view, weather_fg, model) -> pd.DataFrame:
    function_output = getattr(sys.modules[__name__], function["name"])(
        **function["arguments"], feature_view=feature_view, weather_fg=weather_fg, model=model
    )
    if isinstance(function_output, pd.DataFrame):
        col = "grass_pollen" if "grass_pollen" in function_output.columns else "pm25"
        function_output[col] = function_output[col].apply(round, ndigits=2)
    return function_output


def get_context_data(user_query, feature_view, weather_fg, model_pollen, client=None, model_llm=None, tokenizer=None):
    if client:
        # Modern ChatOpenAI Implementation
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=client.api_key)
        prompt = get_function_calling_prompt(user_query).split("<|im_start|>user")[0]
        completion = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_query)]).content
    else:
        # Local LLM Flow
        prompt = get_function_calling_prompt(user_query)
        tokens = tokenizer(prompt, return_tensors="pt").to(model_llm.device)
        generated = model_llm.generate(**tokens, max_new_tokens=256)
        completion = tokenizer.decode(generated.squeeze()[tokens.input_ids.numel() :], skip_special_tokens=True)

    functions = extract_function_calls(completion)
    if functions:
        data = invoke_function(functions[0], feature_view, weather_fg, model_pollen)
        if isinstance(data, pd.DataFrame):
            val_col = "grass_pollen" if "grass_pollen" in data.columns else "pm25"
            return "Pollen Context:\n" + "\n".join([f'Date: {row["date"]}; Level: {row[val_col]}' for _, row in data.iterrows()])
    return ""
