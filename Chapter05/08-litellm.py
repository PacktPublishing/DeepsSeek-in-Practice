# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "litellm",
#     "diskcache"
# ]
# ///


from litellm import completion
from litellm.caching.caching import Cache
import litellm
import os

litellm.cache = Cache(type="disk")

env_var_name = "DEEPSEEK_API_KEY"
assert os.environ[env_var_name], f"Please set your {env_var_name} environment variable."

response = completion(
    model="deepseek/deepseek-reasoner",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of region of Le Marche in Italy?",
        }
    ],
    num_retries=2,  # Number of retries
    fallbacks=["deepseek/deepseek-chat"],  # Fallback models
    caching=True,  # Enable caching
)


final_msg = response.choices[0].message.content
print(f"Final Answer:\n{final_msg}\n")

if hasattr(response.choices[0].message, "reasoning_content"):
    reasoning = response.choices[0].message.reasoning_content
    print(f"Reasoning:\n{reasoning}\n")
