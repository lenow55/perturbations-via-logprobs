from openai import OpenAI

# Настройка клиента
# vLLM по умолчанию работает на порту 8000
client = OpenAI(
    base_url="http://localhost:8996/v1",
    api_key="NYXV2sS3PLDYLbC",  # vLLM обычно не требует реального ключа, можно передать любую строку
)

model_name = "Qwen/Qwen3-8B-FP8"

try:
    print("Запрос отправлен...\n")

    extra_body = {
        "prompt_logprobs": 5,
    }
    # stream = client.chat.completions.create(
    with client.chat.completions.with_streaming_response.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Напиши короткую историю про робота, который любил кофе.",
            }
        ],
        stream=True,  # Включаем стриминг
        temperature=0.7,
        logprobs=True,
        top_logprobs=5,  # Берем топ-5 вариантов для расчета неопределенности
        max_tokens=5,
        # extra_body=extra_body,
        # INFO: со стриммингом prompt_logprobs не включается
    ) as response:
        # Читаем ответ построчно, как он приходит по сети
        # Вы увидите структуру SSE (Server-Sent Events)
        print("--- RAW NETWORK STREAM ---")
        for line in response.iter_lines():
            if line:
                # Декодируем байты в строку для печати
                print(line)
    # # Итерация по полученным чанкам (кусочкам текста)
    # for chunk in stream:
    #     # В режиме стриминга контент находится в delta.content
    #     content = chunk.choices[0].delta.content
    #
    #     if content:
    #         # flush=True заставляет консоль выводить текст мгновенно, не буферизируя
    #         print(content, end="", flush=True)

    print("\n\nГотово.")

except Exception as e:
    print(f"Произошла ошибка: {e}")
