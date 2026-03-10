import json
import logging

from src.metrics import (
    TOKEN_HUB,
    WORD_HUB,
    TokenMetricSignature,
    WordMetricSignature,
)
from src.schemas import PromptLogprob, TokenImportance, WordImportance, WordInfo
from src.utils.words import get_words_and_indices, normalize_text

logger = logging.getLogger(__name__)


class WordAnalyzer:
    _token_metric_func: TokenMetricSignature
    _word_metric_func: WordMetricSignature

    def __init__(
        self,
        token_metric_func: TokenMetricSignature,
        word_metric_func: WordMetricSignature,
    ):
        self._token_metric_func = token_metric_func
        self._word_metric_func = word_metric_func

    def tokens_importances(
        self, prompt_logprobs: list[dict[str, PromptLogprob] | None]
    ) -> list[TokenImportance]:
        data: list[TokenImportance] = []
        for forward in prompt_logprobs:
            if not isinstance(forward, dict):
                continue
            logprobs: list[PromptLogprob] = []
            token_str = None
            for _, logprob in forward.items():
                logprobs.append(logprob)
                if not isinstance(token_str, str):
                    token_str = logprob.decoded_token

            importance = self._token_metric_func(top_logprobs=logprobs)
            data.append({"token": str(token_str), "importance": importance})
        return data

    def words_importances(
        self,
        tokens_importances: list[TokenImportance],
        passage: str,
    ) -> list[WordImportance]:

        data: list[WordImportance] = []
        prompt_buffer: str = ""  # буфер текста
        prompt_tokens_map: list[int] = []  # мапинг текста на id токена
        words_infos: list[WordInfo] = get_words_and_indices(passage)

        counter = 0
        for token_d in tokens_importances:
            prompt_buffer = prompt_buffer + token_d["token"]
            prompt_tokens_map.extend([counter] * len(token_d["token"]))
            counter += 1

        # INFO: обрезаем всё кроме контекста
        start_i = prompt_buffer.find("context: ") + 9
        end_i = prompt_buffer.find("question: ")

        prompt_buffer_c = prompt_buffer[start_i:end_i]
        prompt_tokens_map_c = prompt_tokens_map[start_i:end_i]

        current_pos = 0

        for word in words_infos:
            word_text = word["word"]

            # Пробуем точный поиск
            start_idx = None
            try:
                start_idx = prompt_buffer_c.index(word_text, current_pos)
            except ValueError:
                logger.warning(
                    f"Слово '{word}' не найдено в тексте токенов начиная с позиции {current_pos}."
                )
                data.append(WordImportance(importance=0.0, **word))
                continue

            end_idx = start_idx + len(word_text)

            # Собираем все уникальные токены, которые попали в диапазон слова
            # Используем set для уникальности, затем сортируем
            matched_token_indices = sorted(
                list(set(prompt_tokens_map_c[start_idx:end_idx]))
            )

            # Вычисляем энтропию и нормализуем
            tokens_metrics = [
                tokens_importances[i]["importance"] for i in matched_token_indices
            ]
            word_importance = self._word_metric_func(
                tokens_entropies=tokens_metrics, count_logprobs=len(tokens_metrics)
            )

            data.append(WordImportance(importance=word_importance, **word))

            # ВАЖНО: обновляем позицию для следующего поиска
            current_pos = end_idx

        return data


class WordAnalyzerBuilder:
    _token_metric_func: TokenMetricSignature | None = None
    _word_metric_func: WordMetricSignature | None = None
    metadata: dict[str, str] = {}

    def set_word_metric(self, name: str):
        if name not in WORD_HUB:
            raise ValueError(
                f"metric '{name}' not exist in WORD_HUB: {WORD_HUB.keys()}"
            )
        self._word_metric_func = WORD_HUB[name]
        self.metadata.update({"word_metric": name})

    def set_token_metric(self, name: str):
        if name not in TOKEN_HUB:
            raise ValueError(
                f"metric '{name}' not exist in TOKEN_HUB: {TOKEN_HUB.keys()}"
            )
        self._token_metric_func = TOKEN_HUB[name]
        self.metadata.update({"token_metric": name})

    def build(self) -> WordAnalyzer:
        if not isinstance(self._token_metric_func, TokenMetricSignature):
            raise RuntimeError("token_metric_func not inited")
        if not isinstance(self._word_metric_func, WordMetricSignature):
            raise RuntimeError("word_metric_func not inited")

        obj = WordAnalyzer(
            token_metric_func=self._token_metric_func,
            word_metric_func=self._word_metric_func,
        )
        logger.info(f"Builded WordAnalyzer with metadata: {json.dumps(self.metadata)}")
        return obj
