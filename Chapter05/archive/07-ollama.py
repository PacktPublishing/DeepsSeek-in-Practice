# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ollama",
# ]
# ///


from ollama import chat

messages = [
    {
        "role": "user",
        "content": "What is the capital of Le Marche region in Italy?",
    },
]

response = chat(
    "deepseek-r1:1.5b", messages=messages, think=True, options={"temperature": 0.0}
)

print(f"Thinking:\n========\n\n{response.message.thinking}")
print(f"\nResponse:\n========\n\n{response.message.content}")
