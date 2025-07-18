# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub",
#     "llama-cpp-python",
# ]
# ///


from llama_cpp import Llama

REPO_ID = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
MODEL_FILENAME = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"

llm = Llama.from_pretrained(
    repo_id=REPO_ID,
    filename=MODEL_FILENAME,
    verbose=False,
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "What is the most likely sky color in Copenhagen? Think hard and answer in one word.",
        },
    ],
)

text_response = response["choices"][0]["message"]["content"]

print(f"{MODEL_FILENAME} response:\n{text_response}")
