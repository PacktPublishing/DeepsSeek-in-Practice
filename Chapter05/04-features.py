# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai",
#     "pydantic",
# ]
# ///


from openai import OpenAI
from pydantic import BaseModel, Field
import os
import json
import time

API_KEY = os.getenv("DEEPSEEK_API_KEY")
assert API_KEY, "Please set the DEEPSEEK_API_KEY environment variable."
BASE_URL = "https://api.deepseek.com"
BETA_BASE_URL = "https://api.deepseek.com/beta"
CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL)
BETA_CLIENT = OpenAI(api_key=API_KEY, base_url=BETA_BASE_URL)
TEMPERATURE = 0.0

###################
# Reasoning
###################
model = "deepseek-reasoner"
messages = [
    {"role": "user", "content": "What is the population of Copenhagen in 2030?"}
]
start_time = time.time()
response = CLIENT.chat.completions.create(
    model=model, messages=messages, temperature=TEMPERATURE
)
total_time = time.time() - start_time
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
print("Reasoning Content:")
print(reasoning_content)
print("\nFinal Answer:")
print(content)
print(f"\n\nTotal Time Taken: {total_time:.2f} seconds")
print("-" * 40)


###################
# Streaming
###################
model = "deepseek-chat"
messages = [{"role": "user", "content": "What is the second largest city in Portugal?"}]
start_time = time.time()
total_time = None
response = CLIENT.chat.completions.create(
    model=model, messages=messages, stream=True, temperature=TEMPERATURE
)
for chunk in response:
    if chunk.choices[0].delta.content:
        if total_time is None:
            total_time = time.time() - start_time  # time to first chunk
        print(chunk.choices[0].delta.content, end="", flush=True)
print(f"\n\nTotal time to first chunk: {total_time:.2f} seconds")
print("-" * 40)


###################
# JSON Output
###################
model = "deepseek-chat"
messages = [
    {
        "role": "system",
        "content": "Extract a JSON response with the keys 'name', 'age', and 'city'.",
    },
    {"role": "user", "content": "Duarte is 31 years old and lives in Copenhagen."},
]
start_time = time.time()
response = CLIENT.chat.completions.create(
    model=model,
    messages=messages,
    response_format={"type": "json_object"},
    temperature=TEMPERATURE,
)
total_time = time.time() - start_time
json_object = json.loads(response.choices[0].message.content)
print("JSON Response:")
print(json_object)
print(f"\n\nTotal Time Taken: {total_time:.2f} seconds")
print("-" * 40)

###################
# Function calling
###################
model = "deepseek-chat"


# Create pydantic model for function arguments
class GetPopulationRequest(BaseModel):
    """Request model for fetching population data."""

    city: str = Field(description="The name of the city.")
    year: int = Field(description="The year for which to fetch the population.")


# Define the function to fetch population data
def get_population_for(get_population_request: GetPopulationRequest) -> int:
    """Fetch the population for a given city and year."""
    print(
        f"Fetching population for {get_population_request.city} in {get_population_request.year}..."
    )
    return 1000000  # Dummy data


# Define the tool for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": get_population_for.__name__,
            "description": "Fetch the population for a given city and year.",
            "parameters": {
                "type": "object",
                "properties": GetPopulationRequest.model_json_schema().get(
                    "properties", {}
                ),
                "required": GetPopulationRequest.model_json_schema().get(
                    "required", []
                ),
            },
        },
    },
]

# First message to initiate the conversation
messages = [{"role": "user", "content": "What is the population of Aarhus?"}]
response = CLIENT.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,  # Specify the tools available for function calling
    temperature=TEMPERATURE,
)

# Parse and execute the tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_args = json.loads(tool_call.function.arguments)
tool_response = get_population_for(GetPopulationRequest(**tool_args))

# Append the tool call response to the messages
messages.append(response.choices[0].message)
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(tool_response),
    },
)

# Final message to get the answer after the tool call
response = CLIENT.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMPERATURE,
)

print("Final Answer:")
print(response.choices[0].message.content)
print("-" * 40)


###################
# FIM (Fill in the Middle Completion)
###################
model = "deepseek-chat"

prompt = "The population of Lisbon is exactly "
suffix = " million people."


response = BETA_CLIENT.completions.create(
    model=model,
    prompt=prompt,
    suffix=suffix,
    max_tokens=5,
    temperature=TEMPERATURE,
)

response_text = response.choices[0].text
final_text = prompt + response_text + suffix
print("Final Text:")
print(final_text)
print("-" * 40)
