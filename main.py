# main.py
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI(title="Quantam AI API")

# Load a lightweight text generation model
generator = pipeline("text-generation", model="gpt2")

@app.get("/")
def home():
    return {"message": "Quantam AI API is online!"}

@app.get("/chat")
def chat(prompt: str):
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return {"response": response[0]["generated_text"]}
