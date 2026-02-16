import asyncio
import logging

import numpy as np
from openai import AsyncOpenAI
from pydantic import TypeAdapter

from src.config import ChatLLMConfig
from src.schemas import PromptLogprob, Scenario, ScenarioResult, TokenEntropy
from src.utils.base import calculate_prompt_logprobs

logger = logging.getLogger(__name__)


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


async def analyze_prompt_entropy(
    idx: str,
    scenario: Scenario,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: ChatLLMConfig,
    model: str,
) -> tuple[str, ScenarioResult]:
    """
    Генерирует ответ и возвращает токены и их энтропию.
    """

    _, prompt_logprobs = await calculate_prompt_logprobs(
        idx=idx,
        query=scenario["text"],
        client=client,
        semaphore=semaphore,
        config=config,
        model=model,
    )

    data: list[TokenEntropy] = []
    for forward in prompt_logprobs:
        if not isinstance(forward, dict):
            continue
        logprobs: list[PromptLogprob] = []
        token_str = None
        for _, logprob in forward.items():
            logprobs.append(logprob)
            if not isinstance(token_str, str):
                token_str = logprob.decoded_token

        entropy = calculate_token_entropy(logprobs)
        data.append({"token": str(token_str), "entropy": entropy})

    result = ScenarioResult(logprobs=data, **scenario.copy())

    return idx, result
