import llm_sdk
import src.process_prompt as procp
import src.validation as val
import numpy as np
import src.errors as errors
import src.constrained_decoding as dec
import json


def _is_numeric(param_type: val.ParameterType) -> bool:
    """
    Checks if a parameter type holds a number.

    Args:
        param_type: the type of the parameter.

    Return:
        True for number and integer types, False otherwise.
    """
    return param_type in (val.ParameterType.NUMBER,
                          val.ParameterType.INTEGER)


def _fits_closing(text: str, closing: str) -> bool:
    """
    Checks if the text can lead to the expected closing sequence
    or if it already starts with the complete closing sequence.

    Args:
        text: the text to check.
        closing: the expected closing sequence.

    Return:
        True if the text matches or can lead to the closing sequence,
        False otherwise.
    """
    text = text.lstrip(" ")
    return bool(text) and (closing.startswith(text)
                           or text.startswith(closing))


def _closing_step(tail: str, text: str,
                  closing: str) -> tuple[str, int | None]:
    """
    Adds new text to the current closing sequence and checks if
    the expected closing has been reached.

    Args:
        tail: the text accumulated from previous tokens.
        text: the new text to append.
        closing: the expected closing sequence.

    Return:
        A tuple containing the updated tail and the number of
        characters needed from the new text to complete the closing.
        Returns None as the second element if the closing is incomplete.
    """
    full: str = tail + text
    stripped: str = full.lstrip(" ")
    if stripped.startswith(closing):
        spaces: int = len(full) - len(stripped)
        return full, spaces + len(closing) - len(tail)
    return full, None


def _filter_closing(tail: str, closing: str,
                    vocab: dict[int, str]) -> list[int]:
    """
    Filters the vocabulary to obtain tokens that can continue
    the expected closing sequence.

    Args:
        tail: the text accumulated from previous tokens.
        closing: the expected closing sequence.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        A list containing the IDs of valid tokens.
    """
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if str_tok and _fits_closing(tail + str_tok, closing):
            result_ids.append(id_token)
    return result_ids


def _ends_with_escape(text: str) -> bool:
    """
    Checks if the text ends with an unpaired backslash, which
    indicates that the next character may be escaped.

    Args:
        text: the text to check.

    Return:
        True if the text ends with an unpaired backslash,
        False otherwise.
    """
    count = len(text) - len(text.rstrip("\\"))
    return count % 2 == 1


def _split_closing_quote(token: str,
                         escaped: bool) -> tuple[str, bool]:
    """
    Splits a token at its first unescaped double quote.

    Args:
        token: the token string to inspect.
        escaped: indicates whether the first character is escaped
                 by a backslash from the previous token.

    Return:
        A tuple containing the text before the closing quote and
        a boolean indicating whether an unescaped quote was found.
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
    """
    Decodes JSON escape sequences into their corresponding
    characters.

    Args:
        value: the string containing JSON escape sequences.

    Return:
        The decoded string. Returns the original value if decoding fails.
    """
    try:
        result: str = json.loads(f'"{value}"')
        return result
    except json.JSONDecodeError:
        return value


def _filter_boolean(value: str,
                    vocab: dict[int, str]) -> list[int]:
    """
    Filters the vocabulary to obtain tokens that can continue
    a valid JSON boolean value (true or false).

    Args:
        value: the boolean value generated so far.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        A list containing the IDs of valid tokens.
    """
    possible: list[str] = [pos.removeprefix(value)
                           for pos in ["true", "false"]
                           if pos.startswith(value)]
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if str_tok and any(v.startswith(str_tok) for v in possible):
            result_ids.append(id_token)
    return result_ids


def _filter_number(value: str,
                   closing: str,
                   vocab: dict[int, str],
                   allow_decimal: bool) -> list[int]:
    """
    Filters the vocabulary to obtain tokens that can continue
    a valid JSON number or complete the expected closing sequence.

    Args:
        value: the number generated so far.
        closing: the expected closing sequence after the number.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        A list containing the IDs of valid tokens.
    """
    possible: list[str] = [str(i) for i in range(10)]
    can_end: bool = bool(value) and not value.endswith(("-", "."))
    if not value:
        possible.append("-")
    elif allow_decimal and can_end and "." not in value:
        possible.append(".")
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        tok: str = str_tok if value else str_tok.lstrip(" ")
        if not tok:
            continue
        if can_end and _fits_closing(tok, closing):
            result_ids.append(id_token)
        elif any(v.startswith(tok) for v in possible):
            result_ids.append(id_token)
    return result_ids


def _filter_string(value: str, closing: str,
                   vocab: dict[int, str]) -> list[int]:
    """
    Filters the vocabulary to obtain tokens that can continue
    a valid JSON string, taking escaped characters and closing
    quotes into account.

    Args:
        value: the string content generated so far.
        closing: the expected closing sequence after the string.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        A list containing the IDs of valid tokens.
    """
    escaped: bool = _ends_with_escape(value)
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if not str_tok or "�" in str_tok:
            continue
        content, closes = _split_closing_quote(str_tok, escaped)
        if any(ord(c) < 32 for c in content):
            continue
        rest: str = str_tok[len(content) + 1:]
        if closes and rest and not _fits_closing(rest, closing):
            continue
        result_ids.append(id_token)
    return result_ids


def _valid_tokens_param_filter(value: str,
                               data_type: val.Parameter,
                               vocab: dict[int, str],
                               tail: str | None,
                               closing: str) -> list[int]:
    """
    Selects the appropriate token filter according to the parameter
    type and the current generation state.

    Args:
        value: the parameter value generated so far.
        data_type: the schema defining the parameter type.
        vocab: a dictionary mapping token IDs to their string values.
        tail: the closing sequence accumulated so far, if any.
        closing: the expected closing sequence.

    Return:
        A list containing the IDs of valid tokens for the current step.
    """
    if tail is not None:
        return _filter_closing(tail, closing, vocab)
    if data_type.type == val.ParameterType.BOOLEAN:
        return _filter_boolean(value, vocab)
    elif _is_numeric(data_type.type):
        return _filter_number(
            value, closing, vocab,
            data_type.type == val.ParameterType.NUMBER)
    elif data_type.type == val.ParameterType.STRING:
        return _filter_string(value, closing, vocab)
    raise errors.GenerationError(
        f"Unsupported parameter type: {data_type.type}")


def is_value_complete(data_type: str, value: str) -> bool:
    """
    Checks whether the generated parameter value is complete.

    Args:
        data_type: the type of the parameter.
        value: the parameter value generated so far.

    Return:
        True if the value is a complete boolean (true or false),
        False otherwise.
    """
    return data_type == "boolean" and value in ("true", "false")


def complete_parameters(model: llm_sdk.Small_LLM_Model,
                        res: list[int],
                        prompt: list[int],
                        f_def: list[val.FunctionDefinition],
                        selected_func: str,
                        vocab: dict[int, str]) -> dict[str,
                                                       str |
                                                       int |
                                                       float |
                                                       bool]:

    """
    Generates the parameters of the selected function using
    constrained decoding and the model vocabulary.

    Finds the selected function definition, generates each parameter
    while filtering invalid tokens, and converts the generated
    values into their corresponding Python types.

    Raises errors if the function is not found, no valid tokens
    are available, parameter generation fails, or a generated
    number cannot be converted.

    Args:
        model: the language model used to generate the parameters.
        res: the token IDs generated so far.
        prompt: the initial prompt token IDs.
        f_def: a list containing the available function definitions.
        selected_func: the name of the function whose parameters
                       must be generated.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        A dictionary containing the generated parameter names
        and their values converted to the corresponding Python types.
    """
    function: val.FunctionDefinition | None = next((func
                                                   for func in f_def
                                                   if func.name ==
                                                   selected_func),
                                                   None)
    if function is None:
        raise errors.GenerationError("No function was found, check "
                                     "function_definition.json")
    parameters: dict[str, str | int | float | bool] = {}
    last: int = len(function.parameters) - 1
    for i, (param, data_type) in enumerate(function.parameters.items()):
        if data_type.type == val.ParameterType.STRING:
            res.extend(procp.tokenize_str(model, f'"{param}":"'))
        else:
            res.extend(procp.tokenize_str(model, f'"{param}":'))
        closing: str = "}}" if i == last else ","
        value: str = ""
        tail: str | None = None
        done: bool = False
        steps: int = 0
        while not done and steps < 100:
            steps += 1
            if tail is None and is_value_complete(data_type.type, value):
                tail = ""
            logits: list[float] = model.get_logits_from_input_ids(
                prompt + res)
            valid_tokens: list[int] = _valid_tokens_param_filter(
                value, data_type, vocab, tail, closing)
            if not valid_tokens:
                raise errors.GenerationError(
                    "Response generation error: "
                    "No valid tokens were selected")
            masked_logits = dec.mask_logits(logits, valid_tokens)
            chosen: int = int(np.argmax(masked_logits))
            token_text: str = vocab[chosen]
            start: int = 0
            if tail is None:
                if data_type.type == val.ParameterType.STRING:
                    content, closes = _split_closing_quote(
                        token_text, _ends_with_escape(value))
                    value += content
                    if closes:
                        tail = ""
                        start = len(content) + 1
                elif ((_is_numeric(data_type.type)) and value
                      and _fits_closing(token_text, closing)):
                    tail = ""
                else:
                    value += (token_text if value
                              else token_text.lstrip(" "))
                if tail is None:
                    res.append(chosen)
                    continue
            tail, used = _closing_step(tail, token_text[start:],
                                       closing)
            if used is None:
                res.append(chosen)
            else:
                res.extend(procp.tokenize_str(
                    model, token_text[:start + used]))
                done = True
        if not done:
            raise errors.GenerationError(
                f"Parameter {param} was not closed")
        if data_type.type == val.ParameterType.STRING:
            parameters[param] = _decode_json_string(value)
        elif _is_numeric(data_type.type):
            try:
                if data_type.type == val.ParameterType.INTEGER:
                    parameters[param] = int(value)
                elif data_type.type == val.ParameterType.NUMBER:
                    parameters[param] = float(value)
            except ValueError as err:
                raise errors.CallMeError(
                    f"Invalid number for parameter {param}: "
                    f"{value}.\nError: {err}") from err
        elif data_type.type == val.ParameterType.BOOLEAN:
            parameters[param] = value == "true"
        else:
            raise errors.GenerationError(
                f"Unsupported parameter type: {data_type.type}")
    return parameters
