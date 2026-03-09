from typing import Protocol, runtime_checkable

import numpy as np

from src.schemas import PromptLogprob


@runtime_checkable
class TokenMetricSignature(Protocol):
    def __call__(
        self,
        *,
        top_logprobs: list[PromptLogprob],
    ) -> float: ...


def calculate_token_entropy(top_logprobs: list[PromptLogprob]) -> float:
    """
    Рассчитывает энтропию Шеннона (в битах) на основе Top-K logprobs.
    Формула: H = - sum(p * log2(p))
    """
    probs: list[float] = []
    for item in top_logprobs:
        # OpenAI возвращает logprob (натуральный логарифм), конвертируем в вероятность
        p = np.exp(item.logprob)
        probs.append(p)

    probs_arr = np.array(probs, dtype=float)

    # Нормализуем вероятности, так как у нас только Top-K, а не полный словарь
    # Это дает аппроксимацию энтропии
    probs_norm = probs_arr / np.sum(probs_arr)

    # Считаем энтропию
    entropy = -np.sum(
        probs_norm * np.log2(probs_norm + 1e-9)
    )  # +1e-9 для защиты от log(0)
    if not isinstance(entropy, float):
        raise RuntimeError(f"Bad result type {type(entropy)}")
    return entropy


TOKEN_HUB: dict[str, TokenMetricSignature] = {
    "topk_token_entropy": calculate_token_entropy,
}
