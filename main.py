from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize FastAPI
app = FastAPI(title="Tiny AI Server")

# Load tiny AI model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Request schema
class Prompt(BaseModel):
    text: str

# API route
@app.post("/generate")
def generate(prompt: Prompt):
    inputs = tokenizer(prompt.text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# Health check
@app.get("/")
def root():
    return {"status": "AI server running"}
