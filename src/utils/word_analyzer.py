import json
import logging

from src.metrics import (
    TOKEN_HUB,
    WORD_HUB,
    TokenMetricSignature,
    WordMetricSignature,
)

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
