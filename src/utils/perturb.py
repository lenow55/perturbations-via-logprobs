import asyncio
import logging
import re
from typing import TypedDict

from openai import AsyncOpenAI
from pydantic import TypeAdapter

from src.config import LLMConfig
from src.schemas import PromptLogprob, TokenEntropy, WordInfo, WordInfoRes
from src.utils.base import calculate_token_entropy

logger = logging.getLogger(__name__)


def get_words_and_indices(text: str) -> list[WordInfo]:
    # Паттерн r'\w+' ищет последовательности букв, цифр и нижних подчеркиваний.
    # Это работает и для кириллицы, и для латиницы.
    matches = re.finditer(r"\w+", text)

    results: list[WordInfo] = []

    for match in matches:
        word = match.group()  # Само слово
        start_pos = match.start()  # Начальная позиция (индекс)
        end_pos = match.end()  # Начальная позиция (индекс)

        results.append({"word": word, "start": start_pos, "end": end_pos})

    return results


async def find_ptb_words(
    idx: str,
    scenario: PtbScenario,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: LLMConfig,
    model: str,
) -> tuple[str, PtbScenarioRes]:
    """
    Отсылает запрос в ллм.
    Получает логпробы промпта.
    На основе них ищет слова, которые нужно заменить и отдаёт их позиции
    """
    # INFO: 1. для начала надо разбить вход на слова
    words_infos: list[WordInfo] = get_words_and_indices(scenario["context"])
    text = "context: " + scenario["context"] + "\nquestion: " + scenario["question"]

    # INFO: 2. дальше закидываем запрос в LLM
    async with semaphore:
        logger.debug(f"Start request id {idx}")

        extra_body = {"prompt_logprobs": 5}
        extra_body.update(config.extra_body)

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            logprobs=True,
            top_logprobs=5,  # Берем топ-5 вариантов для расчета неопределенности
            max_tokens=5000,
            extra_body=extra_body,
            **config.params_extra,
        )

    answer = str(response.choices[0].message.content)

    # INFO: 3. Проходим по каждому сгенерированному токену
    ta = TypeAdapter(list[None | dict[str, PromptLogprob]])
    if not response.model_extra:
        raise RuntimeError("Can't compute without prompt logprobs")
    if "prompt_logprobs" not in response.model_extra:
        raise RuntimeError("Can't compute without prompt logprobs")
    prompt_logprobs = ta.validate_python(response.model_extra["prompt_logprobs"])

    entropy2token: list[TokenEntropy] = []
    prompt_buffer: str = ""  # буфер текста
    prompt_tokens_map: list[int] = []  # мапинг текста на id токена

    for i, forward in enumerate(prompt_logprobs):
        if not isinstance(forward, dict):
            continue
        logprobs: list[PromptLogprob] = []
        token_str = None
        for _, logprob in forward.items():
            logprobs.append(logprob)
            if not isinstance(token_str, str):
                token_str = logprob.decoded_token

        entropy = calculate_token_entropy(logprobs)
        entropy2token.append({"token": str(token_str), "entropy": entropy})

        prompt_buffer = prompt_buffer + str(token_str)
        prompt_tokens_map.extend([i] * len(str(token_str)))

    # INFO: обрезаем всё кроме контекста
    start_i = prompt_buffer.find("context: ") + 9
    end_i = prompt_buffer.find("question: ")

    prompt_buffer = prompt_buffer[start_i:end_i]
    prompt_tokens_map = prompt_tokens_map[start_i:end_i]
    res_words: list[WordInfoRes] = []

    current_pos = 0
    for word in words_infos:
        try:
            start_idx = prompt_buffer.index(word["word"], current_pos)
        except ValueError:
            logger.warning(
                f"Слово '{word}' не найдено в тексте токенов начиная с позиции {current_pos}."
            )
            res_words.append(WordInfoRes(entropy=0.0, **word))
            continue

        end_idx = start_idx + len(word["word"])

        # Собираем все уникальные токены, которые попали в диапазон слова
        # Используем set для уникальности, затем сортируем
        matched_token_indices = sorted(list(set(prompt_tokens_map[start_idx:end_idx])))

        # вычисляем энтропию и нормализуем
        word_entropy = float(
            sum([entropy2token[i]["entropy"] for i in matched_token_indices])
        )
        n_word_entropy = word_entropy / len(matched_token_indices)

        res_words.append(WordInfoRes(entropy=n_word_entropy, **word))

    result = PtbScenarioRes(
        words=res_words, answer=answer, logprobs=entropy2token, **scenario.copy()
    )

    return idx, result
