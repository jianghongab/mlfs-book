import xml.etree.ElementTree as ET
import re
import inspect
from typing import get_type_hints, Any, Dict, List
import json
import datetime
import torch
import sys
import pandas as pd

# Import the pollen-specific data retrieval functions
from functions.pollen_data_retrieval import (
    get_historical_pollen_for_date,
    get_historical_pollen_in_date_range,
    get_future_pollen_for_date,
    get_future_pollen_in_date_range,
)


def get_type_name(t: Any) -> str:
    """Get the name of the type."""
    name = str(t)
    return name if "list" in name or "dict" in name else t.__name__


def serialize_function_to_json(func: Any) -> str:
    """Serialize a function to JSON for LLM tool description."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    function_info = {
        "name": func.__name__,
        "description": func.__doc__.strip() if func.__doc__ else "Retrieve pollen data.",
        "parameters": {"type": "object", "properties": {}},
        "returns": "DataFrame",
    }
    for name, _ in signature.parameters.items():
        if name not in ["feature_view", "weather_fg", "model"]:
            function_info["parameters"]["properties"][name] = {"type": "string"}
    return json.dumps(function_info, indent=2)


def extract_function_calls(completion: str) -> List[Dict[str, Any]]:
    """
    Extracts function calls from the LLM completion string.
    Matches XML-style <onefunctioncall> tags and parses the inner JSON.
    """
    completion = completion.strip()
    # Regex to find content inside <onefunctioncall> tags
    pattern = r"(<onefunctioncall>(.*?)</onefunctioncall>)"
    match = re.search(pattern, completion, re.DOTALL)
    if not match:
        return None

    try:
        multiplefn = match.group(1)
        root = ET.fromstring(multiplefn)
        functions = root.findall("functioncall")
        return [json.loads(fn.text) for fn in functions]
    except Exception as e:
        print(f"Error parsing function call: {e}")
        return None


def get_function_calling_prompt(user_query: str) -> str:
    """Prompt template for Stockholm Pollen Expert logic."""
    functions_desc = f"""
{serialize_function_to_json(get_historical_pollen_for_date)}
{serialize_function_to_json(get_historical_pollen_in_date_range)}
{serialize_function_to_json(get_future_pollen_for_date)}
{serialize_function_to_json(get_future_pollen_in_date_range)}
"""
    prompt = f"""<|im_start|>system
You are a Stockholm Pollen Expert. Help users find grass pollen levels.
### FUNCTIONS:
{functions_desc}
### INSTRUCTIONS:
- Use 'get_future_pollen' for forecast/tomorrow queries.
- Use 'get_historical_pollen' for past/yesterday queries.
- Today is {datetime.date.today().strftime("%A")}, {datetime.date.today()}.
Respond STRICTLY with:
<onefunctioncall>
    <functioncall> {{"name": "fn_name", "arguments": {{"arg": "val"}}}} </functioncall>
</onefunctioncall>
<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant"""
    return prompt


def invoke_function(function: Dict[str, Any], feature_view, weather_fg, model) -> pd.DataFrame:
    """Executes the function identified by the LLM."""
    function_name = function["name"]
    arguments = function["arguments"]
    # Dynamically call the function from this module
    function_output = getattr(sys.modules[__name__], function_name)(
        **arguments,
        feature_view=feature_view,
        weather_fg=weather_fg,
        model=model,
    )
    if isinstance(function_output, str):
        return function_output
    # Round target values for both pollen and legacy air quality columns
    target_cols = ["grass_pollen", "pm25", "predicted_pollen"]
    for col in target_cols:
        if col in function_output.columns:
            function_output[col] = function_output[col].apply(round, ndigits=2)
    return function_output


def get_context_data(user_query: str, feature_view, weather_fg, model_pollen, client=None, model_llm=None, tokenizer=None) -> str:
    """Main retrieval logic used by the Streamlit app."""
    if client:
        # OpenAI Flow
        instructions = get_function_calling_prompt(user_query).split("<|im_start|>user")[0]
        completion = (
            client.chat.completions.create(
                model="gpt-3.5-turbo", messages=[{"role": "system", "content": instructions}, {"role": "user", "content": user_query}]
            )
            .choices[0]
            .message.content.strip()
        )
    else:
        # Local LLM Flow
        prompt = get_function_calling_prompt(user_query)
        tokens = tokenizer(prompt, return_tensors="pt").to(model_llm.device)
        generated_tokens = model_llm.generate(**tokens, max_new_tokens=256)
        completion = tokenizer.decode(generated_tokens.squeeze()[tokens.input_ids.numel() :], skip_special_tokens=True)

    functions = extract_function_calls(completion)
    if functions:
        data = invoke_function(functions[0], feature_view, weather_fg, model_pollen)
        if isinstance(data, pd.DataFrame):
            val_col = "grass_pollen" if "grass_pollen" in data.columns else "pm25"
            return "Pollen Context:\n" + "\n".join([f'Date: {row["date"]}; Level: {row[val_col]}' for _, row in data.iterrows()])
    return ""
