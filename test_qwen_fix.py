from httpx import Client
from openai import OpenAI

http_client = Client(proxy="socks5h://localhost:10808")

# Настройка: URL вашего локального сервера vLLM
client = OpenAI(
    api_key="Empty",  # если требуется
    base_url="https://964841c48652.ngrok-free.app/v1",  # по умолчанию vLLM
    http_client=http_client,
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B-FP8",
    messages=[
        {"role": "user", "content": "Привет, как дела?"},
    ],
    max_tokens=50,
    temperature=0.0,
    top_p=1.00,
    logprobs=True,  # добавлено
    top_logprobs=5,  # возвращает пять наиболее вероятных токенов на каждом шаге
    extra_body={
        "prompt_logprobs": 5,
        # "include_reasoning": False, # это проверял на qwen2.5 в landev
        # "reasoning": {"exclude": True},
    },
)

for choice in response.choices:
    # print(f"Message: {choice.message.content}")
    if hasattr(choice, "logprobs") and choice.logprobs:
        if not isinstance(choice.logprobs.content, list):
            continue
        for i, token in enumerate(choice.logprobs.content):
            print(f"Token: {token.token!r} | logprob: {token.logprob}")
            if token.top_logprobs:
                print("  Top candidates:")
                for cand in token.top_logprobs:
                    print(f"    {cand.token!r}: {cand.logprob}")
