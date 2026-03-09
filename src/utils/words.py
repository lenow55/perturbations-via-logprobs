import asyncio
import logging
import re
import unicodedata
from difflib import SequenceMatcher

from openai import AsyncOpenAI

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
from src.utils.metrics_hub import mean_entropy
from src.utils.tokens import calculate_token_entropy

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Нормализует текст для более точного сравнения токенов и слов."""
    # Нормализация Unicode (приводит к единому представлению)
    text = unicodedata.normalize("NFKC", text)
    # Можно добавить дополнительные нормализации при необходимости
    return text


def fuzzy_find(needle: str, haystack: str, start_pos: int = 0) -> int | None:
    """
    Нечёткий поиск подстроки в строке.
    Возвращает позицию начала наиболее похожего участка или None.
    """
    if not needle or not haystack:
        return None

    # Пробуем точный поиск сначала
    try:
        return haystack.index(needle, start_pos)
    except ValueError:
        pass

    # Нечёткий поиск: ищем максимально похожий участок
    best_ratio = 0.0
    best_pos = None
    needle_len = len(needle)

    # Ищем только в разумной окрестности (±50% длины)
    search_len = int(needle_len * 1.5)

    for i in range(
        start_pos, min(len(haystack) - needle_len + 1, start_pos + search_len)
    ):
        substring = haystack[i : i + needle_len]
        ratio = SequenceMatcher(None, needle, substring).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = i

    # Возвращаем позицию только если сходство достаточное (>0.8)
    if best_ratio > 0.8:
        logger.debug(
            f"Нечёткое совпадение для '{needle}': ratio={best_ratio:.2f}, pos={best_pos}"
        )
        return best_pos

    return None


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

    # Нормализуем оба текста для сравнения
    original_context_normalized = normalize_text(scenario["context"])
    prompt_buffer_c_normalized = normalize_text(prompt_buffer_c)

    # Проверяем соответствие текстов
    if original_context_normalized != prompt_buffer_c_normalized:
        logger.warning(
            f"""
Расхождение между исходным контекстом и токенизированным:\n
Исходный: {scenario["context"][:100]}...\n
Токенизированный: {prompt_buffer_c[:100]}..."""
        )

    res_words: list[WordInfoRes] = []
    current_pos = 0

    for word in words_infos:
        word_text = word["word"]
        word_normalized = normalize_text(word_text)

        # Пробуем точный поиск
        start_idx = None
        try:
            start_idx = prompt_buffer_c.index(word_text, current_pos)
        except ValueError:
            # Пробуем нечёткий поиск
            start_idx = fuzzy_find(word_text, prompt_buffer_c, current_pos)

            if start_idx is None:
                logger.warning(
                    f"Слово '{word_text}' не найдено в тексте токенов начиная с позиции {current_pos}. "
                )
                logger.warning(
                    f"Контекст: ...{prompt_buffer_c[max(0, current_pos - 20) : current_pos + 50]}..."
                )
                res_words.append(WordInfoRes(entropy=0.0, **word))
                continue

        end_idx = start_idx + len(word_text)

        # Собираем все уникальные токены, которые попали в диапазон слова
        # Используем set для уникальности, затем сортируем
        matched_token_indices = sorted(
            list(set(prompt_tokens_map_c[start_idx:end_idx]))
        )

        # Вычисляем энтропию и нормализуем
        tokens_entropies = [entropy2token[i]["entropy"] for i in matched_token_indices]
        word_entropy = mean_entropy(
            tokens_entropies=tokens_entropies, count_logprobs=config.count_logprobs
        )

        res_words.append(WordInfoRes(entropy=word_entropy, **word))

        # ВАЖНО: обновляем позицию для следующего поиска
        current_pos = end_idx

    result = PtbScenarioRes(
        words=res_words, answer=answer, logprobs=entropy2token, **scenario.copy()
    )

    return idx, result
