# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "boto3"
# ]
# ///


from openai import OpenAI
import os
import boto3


####################
# CLOUDFLARE
####################

api_key = os.environ["CLOUDFLARE_AUTH_TOKEN"]
account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
assert account_id, "Please set the CLOUDFLARE_ACCOUNT_ID environment variable."
assert api_key, "Please set the CLOUDFLARE_API_KEY environment variable."
model = "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
client = OpenAI(
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
    api_key=api_key,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the most likely sky color in Copenhagen?"},
]
response = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=1000,
    temperature=0.0,
)
raw_response = str(response.choices[0].message.content)
start = raw_response.index("<think>") + len("<think>")
end = raw_response.index("</think>")
thinking_content = raw_response[start:end]
response_text = raw_response[end + len("</think>") :].strip()

print("=" * 10, "Cloudflare Thinking", "=" * 10)
print(thinking_content)
print("=" * 10, "Cloudflare Response", "=" * 10)
print(response_text)


####################
# AWS
####################
brt = boto3.client("bedrock-runtime")
model_id = "us.deepseek.r1-v1:0"

assert os.environ.get("AWS_ACCESS_KEY_ID"), (
    "Please set the AWS_ACCESS_KEY_ID environment variable."
)
assert os.environ.get("AWS_SECRET_ACCESS_KEY"), (
    "Please set the AWS_SECRET_ACCESS_KEY environment variable."
)

conversation = [
    {
        "role": "user",
        "content": [{"text": "What is the most likely sky color in Copenhagen?"}],
    }
]

response = brt.converse(
    modelId=model_id,
    messages=conversation,
    inferenceConfig={"maxTokens": 5000, "temperature": 0.0},
)

response_text = response["output"]["message"]["content"][0]["text"]
response_reasoning = response["output"]["message"]["content"][1]["reasoningContent"][
    "reasoningText"
]["text"]
print("=" * 10, "AWS Thinking", "=" * 10)
print(response_reasoning)
print("=" * 10, "AWS Response", "=" * 10)
print(response_text)


####################
# OpenRouter
####################

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
response = client.chat.completions.create(
    model="deepseek/deepseek-r1-0528",
    messages=[
        {"role": "user", "content": "What is the most likely sky color in Copenhagen?"}
    ],
)

response_reasoning = response.choices[0].message.reasoning
response_text = response.choices[0].message.content

print("=" * 10, "OpenRouter Thinking", "=" * 10)
print(response_reasoning)
print("=" * 10, "OpenRouter Response", "=" * 10)
print(response_text)
