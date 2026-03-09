from collections.abc import Callable
import inspect
from typing import Protocol, TypeVar, Any, runtime_checkable

InputT = TypeVar("InputT", contravariant=True)
ContextT = TypeVar("ContextT", contravariant=True)
ConfigT = TypeVar("ConfigT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


@runtime_checkable
class MainFunctionSignature(Protocol[InputT, ContextT, ConfigT, OutputT]):
    def __call__(
        self,
        *,
        input: InputT,
        context: ContextT,
        config: ConfigT,
    ) -> OutputT: ...


# Дженерик для возвращаемого значения функции
R = TypeVar("R")


def validate_protocol_conformance(
    func: Callable[..., R], protocol_class: type[Any]
) -> bool:
    """
    Проверяет, соответствует ли сигнатура функции методу __call__ протокола.
    Поддерживает проверку имен аргументов в рантайме.
    """
    if not hasattr(protocol_class, "__call__"):
        raise TypeError(f"{protocol_class.__name__} не является вызываемым протоколом.")

    # Получаем ожидаемую сигнатуру из протокола (пропуская 'self')
    proto_sig = inspect.signature(protocol_class.__call__)
    proto_params = list(proto_sig.parameters.values())[1:]  # Убираем self

    # Получаем реальную сигнатуру функции
    func_sig = inspect.signature(func)
    func_params = list(func_sig.parameters.values())

    # 1. Проверка количества аргументов
    if len(proto_params) != len(func_params):
        raise TypeError(
            f"Несоответствие количества аргументов:\nожидалось {len(proto_params)}, получено {len(func_params)}"
        )

    # 2. Проверка имен и типов (порядок важен для позиционных, имена — для keyword-only)
    for p_param, f_param in zip(proto_params, func_params):
        if p_param.name != f_param.name:
            raise NameError(
                f"Ошибка в имени аргумента: ожидалось '{p_param.name}',\nно в функции написано '{f_param.name}'"
            )

    return True


# Теперь всё совпадает символ в символ:
def my_kafka_handler(*, input: str, context: str, confign: dict[str, int]) -> bool:
    print(f"Input: {input}, Context: {context}, Config: {confign}")
    return True


try:
    _ = validate_protocol_conformance(my_kafka_handler, MainFunctionSignature)
except NameError as e:
    print(f"❌ Валидация не прошла: {e}")

# Явное указание типов для проверки
worker: MainFunctionSignature[str, str, dict[str, int], bool] = my_kafka_handler

if isinstance(my_kafka_handler, MainFunctionSignature):
    print("✅ Структурная проверка пройдена")
