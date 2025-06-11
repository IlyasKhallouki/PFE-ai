import requests
import json

# The base URL for your running AI service
BASE_URL = "http://127.0.0.1:8001"

def print_separator():
    """Prints a separator line for cleaner output."""
    print("\n" + "="*50 + "\n")

def test_summarize_endpoint():
    """Tests the /api/v1/summarize endpoint."""
    print("--- 1. Testing Summarization Endpoint ---")
    
    url = f"{BASE_URL}/api/v1/summarize"
    
    # A sample long text to summarize
    long_text = (
        "The team gathered for the weekly sync meeting. "
        "Alice presented the latest marketing metrics, showing a 15% increase in user engagement. "
        "Bob followed up with a technical deep-dive on the new database migration, "
        "confirming that the project is on schedule for a Q3 release. "
        "Charlie raised a concern about server costs, and the team agreed to explore "
        "cost-saving measures and review them in the next meeting. "
        "Finally, the team discussed the upcoming company offsite event, "
        "brainstorming potential locations and activities."
    )
    
    payload = {"text": long_text}
    
    print(f"Original Text:\n'{long_text}'")
    
    try:
        response = requests.post(url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            summary_data = response.json()
            print(f"\nSUCCESS: Received summary.")
            print(f"Summary:\n'{summary_data['summary']}'")
        else:
            print(f"\nERROR: Failed to get summary.")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Connection failed. Is the AI service running?")

def test_chat_endpoint():
    """Tests the /api/v1/chat endpoint with a simulated conversation."""
    print("--- 2. Testing Chat Endpoint (Conversation) ---")
    
    url = f"{BASE_URL}/api/v1/chat"
    
    # --- Conversation Turn 1 ---
    print("\nTurn 1:")
    user_message_1 = "Hello! Can you tell me about Python?"
    payload_1 = {"text": user_message_1}
    
    print(f"User: '{user_message_1}'")
    
    try:
        response_1 = requests.post(url, json=payload_1)
        
        if response_1.status_code != 200:
            print(f"ERROR: Chat request failed.")
            print(f"Status Code: {response_1.status_code}")
            print(f"Response: {response_1.text}")
            return # Stop the test if the first turn fails

        bot_reply_1 = response_1.json()['reply']
        print(f"Bot: '{bot_reply_1}'")

        # --- Conversation Turn 2 (with history) ---
        print("\nTurn 2 (with history):")
        user_message_2 = "What is it most commonly used for?"
        
        payload_2 = {
            "past_user_inputs": [user_message_1],
            "generated_responses": [bot_reply_1],
            "text": user_message_2
        }
        
        print(f"User: '{user_message_2}'")
        
        response_2 = requests.post(url, json=payload_2)
        
        if response_2.status_code == 200:
            bot_reply_2 = response_2.json()['reply']
            print("SUCCESS: Received chat reply with context.")
            print(f"Bot: '{bot_reply_2}'")
        else:
            print(f"ERROR: Chat request failed.")
            print(f"Status Code: {response_2.status_code}")
            print(f"Response: {response_2.text}")

    except requests.exceptions.ConnectionError:
        print("\nERROR: Connection failed. Is the AI service running?")


if __name__ == "__main__":
    print("🚀 Starting AI Service Test Script 🚀")
    
    # First, check if the service is running at all
    try:
        health_check = requests.get(BASE_URL)
        if health_check.status_code == 200:
            print("✅ AI Service is running.")
        else:
            print("❌ AI Service responded with an error.")
    except requests.exceptions.ConnectionError:
        print("❌ FATAL: Could not connect to the AI Service.")
        print("Please ensure 'uvicorn main:app --reload --port 8001' is running.")
    
    print_separator()
    test_summarize_endpoint()
    print_separator()
    test_chat_endpoint()
    print("\n✅ Test script finished.")