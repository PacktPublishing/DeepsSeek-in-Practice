import os

from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI

load_dotenv(".envrc", override=True)


def llm(
    messages: list[dict], model: str, response_format: dict | None = None
) -> tuple[str, str | None]:
    """Call DeepSeek LLM API with messages.

    Args:
        messages: List of chat messages with role and content
        model: Model name (deepseek-chat or deepseek-reasoner)
        response_format: Optional response format specification

    Returns:
        Tuple of (message content, reasoning content if available)
    """
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


def litellm(messages: list[dict]) -> tuple[str, str | None]:
    """Call DeepSeek model through LiteLLM providers.

    Args:
        messages: List of chat messages with role and content

    Returns:
        Tuple of (message content, reasoning content if available)
    """
    response = completion(
        # model="deepseek/deepseek-chat", # Uses DeepSeek API
        # model="bedrock/us.deepseek.r1-v1:0" # Uses AWS bedrock for inference
        model="openrouter/deepseek/deepseek-r1-distill-qwen-14b",  # Uses OpenRouter for inference
        messages=messages,
        temperature=0.0,
    )
    message = response.choices[0].message

    # if there is reasoning content, extract it
    if hasattr(message, "reasoning_content"):
        reasoning_content = message.reasoning_content
    else:
        reasoning_content = None

    return message.content, reasoning_content


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    result, reasoning = litellm(messages)
    print(f"Result: {result}")
    print(f"Reasoning: {reasoning}")
