import os

from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI

load_dotenv(".envrc", override=True)


def llm(
    messages: list[dict], model: str, response_format: dict | None = None
) -> tuple[str, str | None]:
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0.0,
    )
    message = response.choices[0].message

    if hasattr(message, "reasoning_content"):
        reasoning_content = message.reasoning_content
    else:
        reasoning_content = None

    return message.content, reasoning_content


def litellm(
    messages: list[dict], model: str, response_format: dict | None = None
) -> tuple[str, str | None]:
    response = completion(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=0.0,
    )
    message = response.choices[0].message

    if hasattr(message, "reasoning_content"):
        reasoning_content = message.reasoning_content
    else:
        reasoning_content = None

    return message.content, reasoning_content


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
print("=" * 10, "DeepSeek API", "=" * 10)
result, reasoning = llm(messages, "deepseek-chat")
print(f"===Result===\n {result}\n===Reasoning===\n {reasoning}")

print("=" * 10, "Litellm", "=" * 10)
result, reasoning = litellm(messages, "deepseek/deepseek-chat")
print(f"===Result===\n {result}\n===Reasoning===\n {reasoning}")
