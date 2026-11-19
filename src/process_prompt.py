import src.validation as val
import llm_sdk
import json


def tokenize_str(model: llm_sdk.Small_LLM_Model, string: str) -> list[int]:
    tensor = model.encode(string)
    tokens: list[list[int]] = tensor.tolist()
    return tokens[0]


def serialize_prompt(prompt: str) -> str:
    return json.dumps(prompt)


def create_forced_prefix(model: llm_sdk.Small_LLM_Model,
                         prompt: val.PromptInput) -> list[int]:
    prompt_json = serialize_prompt(prompt.prompt)
    default = f'{{"prompt": {prompt_json}, "name": "'
    return tokenize_str(model, default)


def create_function_calling_prompt(model: llm_sdk.Small_LLM_Model,
                                   f_def: list[val.FunctionDefinition],
                                   prompt: val.PromptInput) -> list[int]:
    functions = "\n".join(
        json.dumps(func.model_dump(mode="json"))
        for func in f_def
    )
    prompt_json = serialize_prompt(prompt.prompt)
    secuencial: str = (
        "<|im_start|>system\n"
        "You are a function-calling assistant. "
        "Copy text values exactly as written in the user request, "
        "keeping the same upper and lower case. "
        "Available functions:\n"
        f"<tools>\n{functions}\n</tools>\n"
        "Return a valid JSON object with exactly "
        "the fields prompt, name and parameters:\n"
        f'{{"prompt": {prompt_json}, "name": <function-name>, '
        '"parameters": <args-json-object>}\n'
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    return tokenize_str(model, secuencial)
