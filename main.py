# main.py - AI Service Updates
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

class SmartReplyRequest(BaseModel):
    recent_messages: List[Dict[str, str]] = Field(..., description="List of recent messages with 'author' and 'content' keys")
    current_user: str = Field(..., description="Name of the current user")
    max_suggestions: int = Field(default=3, ge=1, le=5)

class SmartReplyResponse(BaseModel):
    suggestions: List[str] = Field(..., description="List of suggested replies")

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
    # Build a single prompt string with the correct chat template.
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

@app.post("/api/v1/smart-replies", response_model=SmartReplyResponse, tags=["AI Features"])
async def generate_smart_replies(payload: SmartReplyRequest):
    """
    Generate contextual reply suggestions based on recent conversation.
    """
    try:
        # Build conversation context (last 5-10 messages)
        recent_context = []
        for msg in payload.recent_messages[-10:]:  # Last 10 messages for context
            recent_context.append(f"{msg['author']}: {msg['content']}")
        
        conversation_context = "\n".join(recent_context)
        
        # Create prompt for generating reply suggestions
        prompt = f"""<|user|>
Based on this conversation, suggest 3 short, appropriate replies that {payload.current_user} could send. 
Make them natural, contextual, and varied in tone (e.g., one professional, one casual, one question).
Each reply should be on a new line and be concise (under 20 words).

Conversation:
{conversation_context}

Generate 3 reply suggestions for {payload.current_user}:</s>
<|assistant|>"""

        response_data = query_huggingface(
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "return_full_text": False,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "do_sample": True
                }
            },
            API_URL_CHAT
        )
        
        if response_data and isinstance(response_data, list):
            generated_text = response_data[0].get("generated_text", "").strip()
            
            # Parse the suggestions (split by newlines and clean up)
            suggestions = []
            for line in generated_text.split('\n'):
                line = line.strip()
                # Clean up common prefixes
                for prefix in ['1.', '2.', '3.', '-', '*', '•']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                
                if line and len(line) > 3 and len(line) < 100:  # Reasonable length
                    suggestions.append(line)
                    
                if len(suggestions) >= payload.max_suggestions:
                    break
            
            # Fallback suggestions if AI didn't generate good ones
            if not suggestions:
                # Analyze last message for context-aware fallbacks
                last_message = payload.recent_messages[-1] if payload.recent_messages else None
                if last_message:
                    last_content = last_message['content'].lower()
                    if '?' in last_content:
                        suggestions = ["Thanks for asking!", "Let me think about that", "Good question!"]
                    elif 'thanks' in last_content or 'thank you' in last_content:
                        suggestions = ["You're welcome!", "No problem!", "Happy to help!"]
                    elif 'meeting' in last_content:
                        suggestions = ["Sounds good!", "I'll be there", "What time works?"]
                    else:
                        suggestions = ["Got it!", "Makes sense", "Thanks for the update"]
                else:
                    suggestions = ["Hi there!", "Thanks!", "Sounds good!"]
            
            return {"suggestions": suggestions[:payload.max_suggestions]}
            
        else:
            # Fallback suggestions
            return {"suggestions": ["Thanks!", "Got it!", "Sounds good!"]}
            
    except Exception as e:
        print(f"Smart reply error: {e}")
        # Fallback suggestions on error
        return {"suggestions": ["Thanks!", "Got it!", "Let me check on that"]}

@app.get("/", tags=["Health Check"])
async def read_root():
    return {"status": "AI Service is running"}