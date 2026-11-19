import llm_sdk
import src.process_prompt as procp
import src.validation as val
import numpy as np
import src.errors as errors
import src.constrained_decoding as dec


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
    if not value:
        possible.append("-")
    elif value.endswith("-") or value.endswith("."):
        pass
    else:
        possible.extend(["}", ","])
        if "." not in value:
            possible.append(".")
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if str_tok and any(v.startswith(str_tok) for v in possible):
            result_ids.append(id_token)
    return result_ids

def _filter_string(
    value: str,
    vocab: dict[int, str]
) -> list[int]:
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if not str_tok or "�" in str_tok:
            continue
        if str_tok.startswith('"'):
            result_ids.append(id_token)
            continue
        if '"' in str_tok:
            continue
        if any(ord(c) < 32 for c in str_tok):
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
    function: val.FunctionDefinition = next(func
                                            for func in f_def
                                            if func.name == selected_func)
    parameters: dict[str, str | float | bool] = {}
    for i, (param, data_type) in enumerate(function.parameters.items()):
        if data_type.type == "string":
            res.extend(procp.tokenize_str(model, f'"{param}": "'))
        else:
            res.extend(procp.tokenize_str(model, f'"{param}": '))
        value_closed: bool = False
        value: str = ""
        max_tokens: int = 0
        while not value_closed:
            if is_value_complete(data_type.type, value):
                break
            logits: list[float] = model.get_logits_from_input_ids(prompt + res)
            valid_tokens: list[int] = _valid_tokens_param_filter(value, data_type, vocab)
            masked_logits = dec.mask_logits(logits, valid_tokens)
            chosen: int = int(np.argmax(masked_logits))
            if (data_type.type == "string" and vocab[chosen].startswith('"')) or max_tokens == 100:
                value_closed = True
            elif (data_type.type == "number"
                  and vocab[chosen].startswith((",", "}"))):
                value_closed = True
            else:
                value = value + vocab[chosen]
                res.append(chosen)
            max_tokens += 1
            print("---value:---")
            print(value)
        if data_type.type == "string":
            res.extend(procp.tokenize_str(model, '"'))
            parameters[param] = value
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
