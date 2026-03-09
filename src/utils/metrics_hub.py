from typing import Protocol


class EntropyFuncSignature(Protocol):
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


HUB: dict[str, EntropyFuncSignature] = {
    "mean": mean_entropy,
    "first": first_entropy,
}
