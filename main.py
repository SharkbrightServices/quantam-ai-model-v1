from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI(title="Quantam AI API")

# Use a tiny model to save RAM
MODEL_NAME = "distilgpt2"  # ~82M parameters instead of 124M GPT-2

# Load tokenizer and model separately
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Create a lightweight text-generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,            # CPU only
    max_length=50,
    do_sample=True,
)

@app.get("/")
def home():
    return {"message": "Quantam AI API is online!"}

@app.get("/chat")
def chat(prompt: str):
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return {"response": response[0]["generated_text"]}
