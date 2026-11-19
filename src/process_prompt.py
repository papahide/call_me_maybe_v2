import src.validation as val
import llm_sdk
import json
import textwrap


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


def get_system_rules() -> str:
    """Return the instructions for the function-calling model."""
    return textwrap.dedent(r"""
    You are a function-calling assistant. You never reply with prose:
    you turn the user's request into exactly one call to one of the
    available functions.

    1. Choose the function whose description matches what the user
       wants done.
    2. Fill each parameter with the real value the program will
       receive, never a word that describes it. Check: if the program
       received exactly this text, would it do what the user wants?

    Parameter rules:
    - Literal text (inside quotes or in CAPITALS): copy it exactly,
      keeping upper and lower case.
    - A plain noun naming a symbol (not quoted, not in capitals) means
      the symbol itself: write the character once, never the noun,
      singular or plural.
    - number: write only the numeric value.
    - Remove or delete: the replacement is the empty string "".
    - regex: the function runs it as a regular expression. Match every
      item of the kind described, not only the ones in the request:
      use character classes and quantifiers ([...], +, *) instead of
      listing the items you see. A plain word only matches itself.
      Never start or end a pattern with |.
    - In JSON a backslash is doubled: the regex \s+ is "\\s+".

    Example (invented function; use only the functions in <tools>):
    User: Replace every dot in 'a.b.c' with hashes
    Assistant: {"prompt": "Replace every dot in 'a.b.c' with hashes",
    "name": "fn_replace_text",
    "parameters": {"text": "a.b.c", "pattern": "[.]",
    "replacement": "#"}}
    Wrong: "replacement": "hashes". Right: "replacement": "#".
    """).strip() + "\n"


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
        "# Tools\n\n"
        "You are provided with function signatures within "
        "<tools></tools> XML tags:\n"
        f"<tools>\n{functions}\n</tools>\n\n"
        f"{get_system_rules()}\n"
        "Return one json object within <tool_call></tool_call> XML "
        "tags, with exactly the fields prompt, name and parameters:\n"
        "<tool_call>\n"
        f'{{"prompt": {prompt_json}, "name": <function-name>, '
        '"parameters": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
        f"<|im_start|>user\n{prompt.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        "<tool_call>\n"
    )
    return tokenize_str(model, secuencial)