import os
import json
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
from itertools import cycle

app = FastAPI(title="Anthropic API Gateway (Gemini Backend)")

class Message(BaseModel):
    role: str
    content: str

class AnthropicRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    stream: Optional[bool] = False
    system: Optional[str] = None
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)

MODEL_MAPPING = {
    "claude-3-opus-20240229": "gemini-1.5-pro",
    "claude-3-sonnet-20240229": "gemini-1.5-flash",
    "claude-3-haiku-20240307": "gemini-1.5-flash-8b",
    "claude-3-5-sonnet-20241022": "gemini-1.5-pro",
    "claude-3-5-sonnet-20240620": "gemini-1.5-pro",
    "claude-3-5-haiku-20241022": "gemini-1.5-flash",
    "claude-3-opus": "gemini-1.5-pro",
    "claude-3-sonnet": "gemini-1.5-flash",
    "claude-3-haiku": "gemini-1.5-flash-8b",
    "claude-3.5-sonnet": "gemini-1.5-pro",
}

def get_api_keys() -> List[str]:
    """Collect all GEMINI_API_KEY environment variables."""
    keys = []
    base_key = os.getenv("GEMINI_API_KEY")
    if base_key:
        keys.append(base_key)
    
    i = 2
    while True:
        key = os.getenv(f"GEMINI_API_KEY{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY environment variables found")
    
    return keys

api_keys = get_api_keys()
api_key_cycle = cycle(api_keys)

def get_next_api_key() -> str:
    """Get the next API key in rotation."""
    return next(api_key_cycle)

def map_model_name(anthropic_model: str) -> str:
    """Map Anthropic model names to Gemini equivalents."""
    return MODEL_MAPPING.get(anthropic_model, "gemini-1.5-pro")

def convert_messages_to_gemini(messages: List[Message], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Convert Anthropic message format to Gemini format."""
    gemini_messages = []
    
    for msg in messages:
        role = "model" if msg.role == "assistant" else "user"
        gemini_messages.append({
            "role": role,
            "parts": [msg.content]
        })
    
    return gemini_messages

async def stream_gemini_response(
    model_name: str,
    messages: List[Dict[str, str]],
    generation_config: Dict[str, Any],
    system_instruction: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Stream responses from Gemini in Anthropic format."""
    api_key = get_next_api_key()
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )
    
    response = model.generate_content(
        messages,
        generation_config=generation_config,
        stream=True
    )
    
    for chunk in response:
        if chunk.text:
            event_data = {
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": chunk.text
                }
            }
            yield f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n"
    
    final_event = {
        "type": "message_delta",
        "delta": {
            "stop_reason": "end_turn"
        }
    }
    yield f"event: message_delta\ndata: {json.dumps(final_event)}\n\n"

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Anthropic API Gateway (Gemini Backend)",
        "available_api_keys": len(api_keys)
    }

@app.get("/models")
async def list_models():
    """List available models in Anthropic format."""
    models = []
    for anthropic_name, gemini_name in MODEL_MAPPING.items():
        models.append({
            "id": anthropic_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "anthropic",
            "display_name": anthropic_name,
            "backend_model": gemini_name
        })
    
    return {
        "object": "list",
        "data": models
    }

@app.post("/v1/messages")
@app.post("/anthropic")
async def create_message(request: AnthropicRequest):
    """Create a message using Anthropic API format, powered by Gemini."""
    try:
        gemini_model = map_model_name(request.model)
        gemini_messages = convert_messages_to_gemini(request.messages, request.system)
        
        generation_config = {
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p
        if request.top_k is not None:
            generation_config["top_k"] = request.top_k
        
        if request.stream:
            return StreamingResponse(
                stream_gemini_response(
                    gemini_model,
                    gemini_messages,
                    generation_config,
                    request.system
                ),
                media_type="text/event-stream"
            )
        else:
            api_key = get_next_api_key()
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel(
                model_name=gemini_model,
                system_instruction=request.system
            )
            
            response = model.generate_content(
                gemini_messages,
                generation_config=generation_config
            )
            
            return JSONResponse({
                "id": f"msg_{int(time.time())}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response.text
                    }
                ],
                "model": request.model,
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
                }
            })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions in Anthropic-compatible format."""
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": str(exc)
            }
        }
    )
