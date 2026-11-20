import llm_sdk
import src.parse as parse
from typing import Any


def _load_vocab(model: llm_sdk.Small_LLM_Model) -> dict[str, int]:
    """
    Acceses the vocabulary file containing the original vocabulary of
    the LLM model.

    Args:
        model: instance of the LLM model.
    Return:
        Adictionary containing the original vocabulary that contains:
        key: Token Value
        value: Token ID
    """
    content: str = parse.read_file(model.get_path_to_vocab_file())
    json_content: Any = parse.parse_json_file(content)
    return parse.validate_vocab(json_content)


def create_vocab(model: llm_sdk.Small_LLM_Model) -> dict[int, str]:
    """
    Creates a usable vocabulary from the original LLm vocabulary.

    Args:
        model: LLM instance that contains the needed functions to create
               the vocab.
    Return:
        Dictionary that contains the LLM vocabulary
        key: Token ID
        value: Token Value.
    """
    vocab: dict[str, int] = _load_vocab(model)
    return {token_id: model.decode([token_id]) for token_id in vocab.values()}
