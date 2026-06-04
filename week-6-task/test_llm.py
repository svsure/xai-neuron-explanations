from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

response = generator(
    "What is the capital of France?",
    max_new_tokens=20,
    do_sample=False
)

print(response[0]["generated_text"])