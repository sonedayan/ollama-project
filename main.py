# import os
# from ollama import Client
# from dotenv import load_dotenv

# load_dotenv()

# # Cloud endpoint + your API key
# client = Client(
#     host="https://ollama.com",  # Ollama Cloud base URL
#     headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
# )

# messages = [
#     {"role": "user", "content": "Why is the sky blue? Explain in simple terms."}
# ]

# # Now it works! (Use a cloud model like gpt-oss:120b-cloud)
# for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
#     print(part['message']['content'], end='', flush=True)
# print()  # New line at end

# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from wrapper import OllamaChat
import json
from dotenv import load_dotenv



load_dotenv()

app = FastAPI(title="Ollama FastAPI Streaming Chat", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← Allows any origin (file://, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],        # ← Allows OPTIONS, POST, GET, etc.
    allow_headers=["*"],        # ← Allows Content-Type, Authorization, etc.
)

# Change model here once
chat = OllamaChat(model="gpt-oss:120b-cloud")   # ← or "gemma2:2b", "llama3.2:3b", etc.

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatRequest):
    messages = [m.dict() for m in request.messages]

    def event_stream():
        full_text = ""
        for part in chat.stream_completion(messages):
            if isinstance(part, str):
                full_text += part
                # Send Server-Sent Events (SSE) format
                data = {
                    "choices": [{"delta": {"content": part}}]
                }
                yield f"data: {json.dumps(data)}\n\n"
            else:
                # Final message with usage
                data = {
                    "choices": [{"finish_reason": "stop", "message": {"content": full_text}}],
                    "usage": {
                        "prompt_tokens": part["input_tokens"],
                        "completion_tokens": part["output_tokens"],
                        "total_tokens": part["total_tokens"]
                    }
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/")
async def root():
    return {"message": "Ollama is running! Use POST /v1/chat/completions"}