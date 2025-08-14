from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
device = "mps"
question = "What is the capital of Le Marche, Italy?"
max_tokens = 1200

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

messages = [
    {"role": "user", "content": question},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=max_tokens)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
