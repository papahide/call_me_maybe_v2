import llm_sdk
import src.process_prompt as procp
import src.validation as val
import numpy as np
import src.errors as errors
import src.constrained_decoding as dec
import json


def _is_number_end(token: str) -> bool:
    """Tell if a token closes a number value."""
    return token.startswith((",", "}", " ", "\n"))


def _ends_with_escape(text: str) -> bool:
    """Tell if text ends with an unpaired backslash."""
    count = len(text) - len(text.rstrip("\\"))
    return count % 2 == 1


def _split_closing_quote(token: str,
                         escaped: bool) -> tuple[str, bool]:
    """Split a token at its first unescaped quote.

    Returns the string content before the quote and whether the
    token closes the string. escaped means the previous token
    ended with a backslash, so the first char is escaped.
    """
    i = 1 if escaped else 0
    while i < len(token):
        if token[i] == "\\":
            i += 2
            continue
        if token[i] == '"':
            return token[:i], True
        i += 1
    return token, False


def _decode_json_string(value: str) -> str:
    """Turn JSON escapes (\\\\, \\") into real characters."""
    try:
        result: str = json.loads(f'"{value}"')
        return result
    except json.JSONDecodeError:
        return value


def _filter_boolean(value: str,
                    vocab: dict[int, str]) -> list[int]:
    possible: list[str] = [pos.removeprefix(value)
                           for pos in ["true", "false"]
                           if pos.startswith(value)]
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if str_tok and any(v.startswith(str_tok) for v in possible):
            result_ids.append(id_token)
    return result_ids


def _filter_number(value: str,
                   vocab: dict[int, str]) -> list[int]:
    possible: list[str] = [str(i) for i in range(10)]
    can_end: bool = bool(value) and not value.endswith(("-", "."))
    if not value:
        possible.append("-")
    elif can_end and "." not in value:
        possible.append(".")
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if not str_tok:
            continue
        if can_end and _is_number_end(str_tok):
            result_ids.append(id_token)
        elif any(v.startswith(str_tok) for v in possible):
            result_ids.append(id_token)
    return result_ids


def _filter_string(value: str,
                   vocab: dict[int, str]) -> list[int]:
    escaped: bool = _ends_with_escape(value)
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if not str_tok or "�" in str_tok:
            continue
        content, _ = _split_closing_quote(str_tok, escaped)
        if any(ord(c) < 32 for c in content):
            continue
        result_ids.append(id_token)
    return result_ids


def _valid_tokens_param_filter(value: str,
                               data_type: val.Parameter,
                               vocab: dict[int, str]) -> list[int]:
    if data_type.type == "boolean":
        return _filter_boolean(value, vocab)
    elif data_type.type == "number":
        return _filter_number(value, vocab)
    else:
        return _filter_string(value, vocab)


def is_value_complete(data_type: str, value: str) -> bool:
    return data_type == "boolean" and value in ("true", "false")


def complete_parameters(model: llm_sdk.Small_LLM_Model,
                        res: list[int],
                        prompt: list[int],
                        f_def: list[val.FunctionDefinition],
                        selected_func: str,
                        vocab: dict[int, str]) -> dict[str,
                                                       str | float | bool]:
    function: val.FunctionDefinition | None = next((func
                                                    for func in f_def
                                                    if func.name ==
                                                    selected_func),
                                                    None)
    if function is None:
        raise errors.GenerationError("No function was found, check "
                                     "function_definition.json")
    parameters: dict[str, str | float | bool] = {}
    for i, (param, data_type) in enumerate(function.parameters.items()):
        if data_type.type == "string":
            res.extend(procp.tokenize_str(model, f'"{param}": "'))
        else:
            res.extend(procp.tokenize_str(model, f'"{param}": '))
        value_closed: bool = False
        value: str = ""
        max_tokens: int = 0
        while not value_closed and max_tokens < 100:
            if is_value_complete(data_type.type, value):
                break
            logits: list[float] = model.get_logits_from_input_ids(
                prompt + res)
            valid_tokens: list[int] = _valid_tokens_param_filter(
                value, data_type, vocab)
            if not valid_tokens:
                raise errors.GenerationError(f"Response generation error: "
                                             "No valid tokens were selected")
            masked_logits = dec.mask_logits(logits, valid_tokens)
            chosen: int = int(np.argmax(masked_logits))
            token_text: str = vocab[chosen]
            if data_type.type == "string":
                content, closes = _split_closing_quote(
                    token_text, _ends_with_escape(value))
                value += content
                if not closes:
                    res.append(chosen)
                else:
                    if content:
                        res.extend(procp.tokenize_str(model, content))
                    value_closed = True
            elif (data_type.type == "number"
                  and _is_number_end(token_text)):
                value_closed = True
            else:
                value += token_text
                res.append(chosen)
            max_tokens += 1
        if data_type.type == "string":
            res.extend(procp.tokenize_str(model, '"'))
            parameters[param] = _decode_json_string(value)
        elif data_type.type == "number":
            try:
                parameters[param] = float(value)
            except ValueError as err:
                raise errors.CallMeError("Invalid number for "
                                         f"parameter {param}: {value}. \n"
                                         f"Error: {err}") from err
        elif data_type.type == "boolean":
            parameters[param] = value == "true"
        if i < len(function.parameters) - 1:
            res.extend(procp.tokenize_str(model, ", "))
        else:
            res.extend(procp.tokenize_str(model, "}"))
    return parameters
