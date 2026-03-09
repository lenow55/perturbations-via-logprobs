from src.utils.metrics_hub import WORD_HUB, WordEntropyFuncSignature


class WordEntropyBuilder:
    token_metric_func: None = None
    word_metric_func: WordEntropyFuncSignature | None = None
    metadata: dict[str, str] = {}

    def set_word_metric(self, name: str):
        if name not in WORD_HUB:
            raise ValueError(
                f"metric '{name}' not exist in WORD_HUB: {WORD_HUB.keys()}"
            )
        self.word_metric_func = WORD_HUB[name]

    def set_word_metric(self, name: str):
        if name not in WORD_HUB:
            raise ValueError(
                f"metric '{name}' not exist in WORD_HUB: {WORD_HUB.keys()}"
            )
        self.word_metric_func = WORD_HUB[name]

    def build(self):
        pass
