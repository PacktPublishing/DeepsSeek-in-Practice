# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "tenacity",
# ]
# ///


from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
import os


API_KEY = os.getenv("DEEPSEEK_API_KEY")
assert API_KEY, "Please set the DEEPSEEK_API_KEY environment variable."
BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def make_request() -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of Portugal?"},
        ],
        stream=False,
        max_tokens=100,
    )
    response_text = response.choices[0].message.content
    if not response_text:
        raise ValueError("Received empty response from the API.")
    return response_text


print(make_request())
