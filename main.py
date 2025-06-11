# main.py
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set HF_API_TOKEN in your .env file.")

API_URL_SUMMARIZE = "https://api-inference.huggingface.co/models/philschmid/bart-large-cnn-samsum"
API_URL_CHAT = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- Pydantic Models ---
class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50)

class SummarizeResponse(BaseModel):
    summary: str

class ChatRequest(BaseModel):
    past_user_inputs: List[str] = Field(default_factory=list)
    generated_responses: List[str] = Field(default_factory=list)
    text: str = Field(...)

class ChatResponse(BaseModel):
    reply: str

# --- FastAPI Application ---
app = FastAPI(
    title="Chat Application AI Service",
    description="A microservice to handle AI tasks.",
    version="1.0.0"
)

# --- Helper Function ---
def query_huggingface(payload: dict, api_url: str) -> dict:
    response = requests.post(api_url, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error from Hugging Face API: {response.text}"
        )
    return response.json()

# --- API Endpoints ---
@app.post("/api/v1/summarize", response_model=SummarizeResponse, tags=["AI Features"])
async def summarize(payload: SummarizeRequest):
    try:
        response_data = query_huggingface(
            {"inputs": payload.text, "parameters": {"min_length": 10, "max_length": 50}},
            API_URL_SUMMARIZE
        )
        summary = response_data[0].get("summary_text", "Sorry, I could not generate a summary.")
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["AI Features"])
async def chat(payload: ChatRequest):
    """
    Generates a conversational reply using a single, formatted prompt string.
    """
    # ***[FIX]*** Build a single prompt string with the correct chat template.
    prompt_parts = []
    # Interleave past user inputs and generated responses
    for user_input, assistant_response in zip(payload.past_user_inputs, payload.generated_responses):
        prompt_parts.append(f"<|user|>\n{user_input}</s>")
        prompt_parts.append(f"<|assistant|>\n{assistant_response}</s>")

    # Add the latest user message
    prompt_parts.append(f"<|user|>\n{payload.text}</s>")
    # Add the token that prompts the assistant to start talking
    prompt_parts.append("<|assistant|>")

    # Join all parts into a single string
    prompt = "\n".join(prompt_parts)

    try:
        response_data = query_huggingface(
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "return_full_text": False, # We only want the new reply
                    "temperature": 0.7,
                    "top_p": 0.95
                }
            },
            API_URL_CHAT
        )
        
        if response_data and isinstance(response_data, list):
            reply = response_data[0].get("generated_text", "").strip()
        else:
            reply = "I am not sure how to respond to that."

        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "AI Service is running"}