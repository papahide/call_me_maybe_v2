import llm_sdk
import src.validation as val
import numpy as np
import src.errors as errors
import src.constrained_decoding as dec


def _valid_tokens_name_filter(name: str,
                              func_names: list[str],
                              vocab: dict[int, str]) -> list[int]:
    """
    Filters the vocabulary to obtain tokens that can continue
    a valid function name or close the function name with a quote.

    Args:
        name: the function name generated so far.
        func_names: a list containing the available function names.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        A list containing the IDs of valid tokens.
    """
    possible: list[str] = [pos.removeprefix(name)
                           for pos in func_names
                           if pos.startswith(name)]
    result_ids: list[int] = []
    for id_token, str_tok in vocab.items():
        if (str_tok and any(v.startswith(str_tok) for v in possible)) \
             or (name in func_names and str_tok.startswith('"')):
            result_ids.append(id_token)
    return result_ids


def complete_name(model: llm_sdk.Small_LLM_Model,
                  res: list[int],
                  prompt: list[int],
                  f_def: list[val.FunctionDefinition],
                  vocab: dict[int, str]) -> str:

    """
    Generates the name of a function using constrained decoding
    and the model vocabulary.

    Extracts the available function names from the definitions,
    filters invalid tokens at each generation step, and generates
    the function name until a closing quote is found.

    Raises an error if no valid tokens are available during
    function name generation.

    Args:
        model: the language model used to generate the function name.
        res: the token IDs generated so far.
        prompt: the initial prompt token IDs.
        f_def: a list containing the available function definitions.
        vocab: a dictionary mapping token IDs to their string values.

    Return:
        The generated function name as a string.
    """
    func_names: list[str] = [func.name for func in f_def]
    name_closed: bool = False
    name: str = ""
    while not name_closed:
        logits: list[float] = model.get_logits_from_input_ids(prompt + res)
        valid_tokens: list[int] = _valid_tokens_name_filter(name,
                                                            func_names,
                                                            vocab)
        if not valid_tokens:
            raise errors.GenerationError("Error generating function name: "
                                         "no valid token was found.")
        masked_logits: np.ndarray = dec.mask_logits(logits, valid_tokens)
        chosen: int = int(np.argmax(masked_logits))
        if name in func_names and vocab[chosen].startswith('"'):
            name_closed = True
        else:
            name += vocab[chosen]
            res.append(chosen)
    return name
