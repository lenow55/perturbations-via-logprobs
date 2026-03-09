from typing import Protocol, runtime_checkable


@runtime_checkable
class WordMetricSignature(Protocol):
    def __call__(
        self,
        *,
        tokens_entropies: list[float],
        count_logprobs: int,
    ) -> float: ...


def mean_entropy(tokens_entropies: list[float], count_logprobs: int):
    word_entropy = float(sum(tokens_entropies))
    n_word_entropy = word_entropy / len(tokens_entropies)
    return n_word_entropy


def first_entropy(tokens_entropies: list[float], count_logprobs: int):
    try:
        word_entropy = tokens_entropies[0]
        return word_entropy
    except IndexError:
        return 0.0


WORD_HUB: dict[str, WordMetricSignature] = {
    "mean": mean_entropy,
    "first": first_entropy,
}
