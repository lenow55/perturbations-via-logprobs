import asyncio
import logging

from openai import AsyncOpenAI

from src.config import ChatLLMConfig
from src.metrics.tokens import calculate_token_entropy
from src.schemas import PromptLogprob, Scenario, ScenarioResult, TokenEntropy
from src.utils.base import calculate_prompt_logprobs

logger = logging.getLogger(__name__)


async def analyze_prompt_entropy(
    idx: str,
    scenario: Scenario,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: ChatLLMConfig,
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
