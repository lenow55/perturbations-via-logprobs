import asyncio
import logging
import re

from openai import AsyncOpenAI
from pydantic import TypeAdapter

from src.config import LLMConfig
from src.schemas import (
    PromptLogprob,
    PtbScenario,
    PtbScenarioRes,
    TokenEntropy,
    WordInfo,
    WordInfoRes,
)
from src.utils.base import calculate_prompt_logprobs
from src.utils.tokens import calculate_token_entropy

logger = logging.getLogger(__name__)


def get_words_and_indices(text: str) -> list[WordInfo]:
    # Разбор паттерна:
    # [«"'({\[]* - Ноль или более открывающих скобок или кавычек любых типов
    # \w+          - Первая часть слова (буквы, цифры)
    # (?:[-']\w+)* - Опциональные продолжения слова через дефис или апостроф (например, "don't" или "кто-то")
    # [»"')}\]]* - Ноль или более закрывающих скобок или кавычек
    pattern = r"[«\"'({\[]*\w+(?:[-']\w+)*[»\"')}\]]*"

    matches = re.finditer(pattern, text)

    results: list[WordInfo] = []

    for match in matches:
        results.append(
            {"word": match.group(), "start": match.start(), "end": match.end()}
        )

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

    answer, prompt_logprobs = await calculate_prompt_logprobs(
        idx=idx,
        query=text,
        client=client,
        semaphore=semaphore,
        config=config,
        model=model,
    )

    entropy2token: list[TokenEntropy] = []
    prompt_buffer: str = ""  # буфер текста
    prompt_tokens_map: list[int] = []  # мапинг текста на id токена

    counter = 0
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
        entropy2token.append({"token": str(token_str), "entropy": entropy})

        prompt_buffer = prompt_buffer + str(token_str)
        prompt_tokens_map.extend([counter] * len(str(token_str)))
        counter += 1

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
