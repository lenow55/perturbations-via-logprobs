import asyncio
import json
import logging
from logging import config as log_config_m

import numpy as np
from httpx import AsyncClient, Timeout
from openai import AsyncOpenAI
from pydantic import TypeAdapter

from src.config import AppSettings, LLMConfig
from src.schemas import PromptLogprob, Scenario, ScenarioResult, TokenEntropy

logger = logging.getLogger(__name__)


def configure_logging(config: AppSettings):
    with open(config.logging_conf_file) as l_f:
        logging_config_dict = json.loads(l_f.read())
        log_config_m.dictConfig(logging_config_dict)


def create_openai_client(config: LLMConfig) -> AsyncOpenAI:
    if config.proxy_url:
        http_client = AsyncClient(proxy="socks5h://localhost:10808")
    else:
        http_client = AsyncClient()

    client = AsyncOpenAI(
        api_key=config.api_key.get_secret_value(),
        base_url=config.base_url,
        timeout=Timeout(config.timeout),
        http_client=http_client,
    )
    return client


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
    config: LLMConfig,
    model: str,
) -> tuple[str, ScenarioResult]:
    """
    Генерирует ответ и возвращает токены и их энтропию.
    """
    async with semaphore:
        logger.debug(f"Start request id {idx}")

        extra_body = {"prompt_logprobs": 5}
        extra_body.update(config.extra_body)

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": scenario["text"]}],
            logprobs=True,
            top_logprobs=5,  # Берем топ-5 вариантов для расчета неопределенности
            max_tokens=5,
            extra_body=extra_body,
            **config.params_extra,
        )

    # Проходим по каждому сгенерированному токену
    ta = TypeAdapter(list[None | dict[str, PromptLogprob]])
    if not response.model_extra:
        raise RuntimeError("Can't compute without prompt logprobs")
    if "prompt_logprobs" not in response.model_extra:
        raise RuntimeError("Can't compute without prompt logprobs")
    prompt_logprobs = ta.validate_python(response.model_extra["prompt_logprobs"])

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
