import numpy as np


def mask_logits(logits: list[float],
                 valid_tokens: list[int]) -> np.ndarray:
    logits_array: np.ndarray = np.array(logits)
    result: np.ndarray = np.full(len(logits), -np.inf)
    result[valid_tokens] = logits_array[valid_tokens]
    return result