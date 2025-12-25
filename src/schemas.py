from typing import TypedDict
from pydantic import BaseModel, Field


class TokenEntropy(TypedDict):
    token: str
    entropy: float


class Scenario(TypedDict):
    text: str
    name: str


class ScenarioResult(Scenario):
    logprobs: list[TokenEntropy]


class PromptLogprob(BaseModel):
    decoded_token: str
    """The token."""

    logprob: float
    """The log probability of this token, if it is within the top 20 most likely
    tokens.

    Otherwise, the value `-9999.0` is used to signify that the token is very
    unlikely.
    """

    rank: int
    """
    Позиция в отранжированном списке токенов
    """


class Answer(BaseModel):
    """
    Модель отдельного варианта ответа.
    """

    idx: int = Field(
        ...,
        description="Уникальный числовой идентификатор варианта ответа (в рамках одного вопроса).",
    )
    text: str = Field(..., description="Текст варианта ответа.")
    label: int = Field(
        ..., description="Метка правильности ответа: 1 — верный ответ, 0 — неверный."
    )


class Question(BaseModel):
    """
    Модель вопроса к тексту с вариантами ответов.
    """

    idx: int = Field(
        ...,
        description="Уникальный числовой идентификатор вопроса (в рамках одного текста).",
    )
    question: str = Field(..., description="Текст вопроса, задаваемого к отрывку.")
    answers: list[Answer] = Field(
        ..., description="Список предложенных вариантов ответов для данного вопроса."
    )


class PassageData(BaseModel):
    """
    Контейнер для текста и связанных с ним вопросов.
    """

    text: str = Field(
        ...,
        description="Основной текст (отрывок) для чтения, содержащий информацию для ответов.",
    )
    questions: list[Question] = Field(
        ..., description="Список вопросов, относящихся к данному тексту."
    )


class ReadingComprehensionItem(BaseModel):
    """
    Корневая модель для одного элемента датасета.
    """

    idx: int = Field(
        ..., description="Глобальный уникальный идентификатор записи в датасете."
    )
    passage: PassageData = Field(
        ..., description="Объект, содержащий текст и структуру вопросов-ответов."
    )
