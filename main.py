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
    allow_origins=["*"],        
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],     
)

# Change the model name as needed
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