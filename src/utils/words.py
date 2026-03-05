import asyncio
import logging
import re

from openai import AsyncOpenAI
from pydantic import TypeAdapter

from src.config import ChatLLMConfig
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


def mean_entropy(tokens_entropies: list[float], count_logprobs: int):
    word_entropy = float(sum(tokens_entropies))
    n_word_entropy = word_entropy / len(tokens_entropies)
    return n_word_entropy


async def find_ptb_words(
    idx: str,
    scenario: PtbScenario,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: ChatLLMConfig,
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

    prompt_buffer_c = prompt_buffer[start_i:end_i]
    prompt_tokens_map_c = prompt_tokens_map[start_i:end_i]
    res_words: list[WordInfoRes] = []

    current_pos = 0
    for word in words_infos:
        try:
            start_idx = prompt_buffer_c.index(word["word"], current_pos)
        except ValueError:
            logger.warning(
                f"Слово '{word}' не найдено в тексте токенов начиная с позиции {current_pos}."
            )
            res_words.append(WordInfoRes(entropy=0.0, **word))
            continue

        end_idx = start_idx + len(word["word"])

        # Собираем все уникальные токены, которые попали в диапазон слова
        # Используем set для уникальности, затем сортируем
        matched_token_indices = sorted(
            list(set(prompt_tokens_map_c[start_idx:end_idx]))
        )

        # вычисляем энтропию и нормализуем
        tokens_entropies = [entropy2token[i]["entropy"] for i in matched_token_indices]
        word_entropy = mean_entropy(
            tokens_entropies=tokens_entropies, count_logprobs=config.count_logprobs
        )

        res_words.append(WordInfoRes(entropy=word_entropy, **word))

    result = PtbScenarioRes(
        words=res_words, answer=answer, logprobs=entropy2token, **scenario.copy()
    )

    return idx, result
