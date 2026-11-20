import numpy as np


def mask_logits(logits: list[float],
                valid_tokens: list[int]) -> np.ndarray:
    """
    Masks invalid tokens by assigning them negative infinity.

    Args:
        logits: the model logits.
        valid_tokens: the IDs of the allowed tokens.

    Return:
        An array with only valid token logits preserved.
    """
    logits_array: np.ndarray = np.array(logits)
    result: np.ndarray = np.full(len(logits), -np.inf)
    result[valid_tokens] = logits_array[valid_tokens]
    return result
