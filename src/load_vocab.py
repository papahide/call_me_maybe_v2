import llm_sdk
import src.parse as parse


def _load_vocab(model: llm_sdk.Small_LLM_Model) -> dict[str, int]:
    content: str = parse.read_file(model.get_path_to_vocab_file())
    return parse.parse_json_file(content)


def create_vocab(model: llm_sdk.Small_LLM_Model) -> dict[int, str]:
    vocab: dict[str, int] = _load_vocab(model)
    return {token_id: model.decode([token_id]) for token_id in vocab.values()}