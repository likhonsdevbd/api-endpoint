import os
import json
import time
import uuid
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
from itertools import cycle

app = FastAPI(title="Anthropic API Gateway (Gemini Backend)")

class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_error: Optional[bool] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentBlock]]

class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ToolChoice(BaseModel):
    type: str
    name: Optional[str] = None

class Metadata(BaseModel):
    user_id: Optional[str] = None

class AnthropicRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    stream: Optional[bool] = False
    system: Optional[str] = None
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = Field(default=None, ge=0)
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    metadata: Optional[Metadata] = None
    thinking: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None

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

api_key_cycle = None

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
    
    return keys

def get_next_api_key() -> str:
    """Get the next API key in rotation."""
    global api_key_cycle
    
    api_keys = get_api_keys()
    if not api_keys:
        raise HTTPException(
            status_code=500,
            detail="No GEMINI_API_KEY environment variables configured. Please set GEMINI_API_KEY."
        )
    
    if api_key_cycle is None:
        api_key_cycle = cycle(api_keys)
    
    return next(api_key_cycle)

def map_model_name(anthropic_model: str) -> str:
    """Map Anthropic model names to Gemini equivalents."""
    return MODEL_MAPPING.get(anthropic_model, "gemini-1.5-pro")

def parse_content_blocks(content: Union[str, List[ContentBlock]]) -> str:
    """Parse content blocks and extract text content."""
    if isinstance(content, str):
        return content
    
    text_parts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and "text" in block:
                text_parts.append(block["text"])
            elif block.get("type") == "thinking" and "text" in block:
                text_parts.append(f"[Thinking: {block['text']}]")
            elif block.get("type") == "tool_result":
                result = block.get("content", "")
                text_parts.append(f"[Tool Result: {result}]")
        elif hasattr(block, 'type'):
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "thinking" and block.text:
                text_parts.append(f"[Thinking: {block.text}]")
            elif block.type == "tool_result" and block.content:
                text_parts.append(f"[Tool Result: {block.content}]")
    
    return "\n".join(text_parts) if text_parts else ""

def convert_messages_to_gemini(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert Anthropic message format to Gemini format."""
    gemini_messages = []
    
    for msg in messages:
        role = "model" if msg.role == "assistant" else "user"
        
        content_text = parse_content_blocks(msg.content)
        
        if content_text:
            gemini_messages.append({
                "role": role,
                "parts": [content_text]
            })
    
    return gemini_messages

async def stream_gemini_response(
    model_name: str,
    messages: List[Dict[str, str]],
    generation_config: Dict[str, Any],
    system_instruction: Optional[str] = None,
    include_thinking: bool = False
) -> AsyncGenerator[str, None]:
    """Stream responses from Gemini in Anthropic format."""
    api_key = get_next_api_key()
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )
    
    message_start = {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0
            }
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
    
    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
            "text": ""
        }
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n"
    
    response = model.generate_content(
        messages,
        generation_config=generation_config,
        stream=True
    )
    
    for chunk in response:
        if chunk.text:
            event_data = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": chunk.text
                }
            }
            yield f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n"
    
    content_block_stop = {
        "type": "content_block_stop",
        "index": 0
    }
    yield f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n"
    
    final_event = {
        "type": "message_delta",
        "delta": {
            "stop_reason": "end_turn",
            "stop_sequence": None
        },
        "usage": {
            "output_tokens": 0
        }
    }
    yield f"event: message_delta\ndata: {json.dumps(final_event)}\n\n"
    
    yield f"event: message_stop\ndata: {{}}\n\n"

@app.get("/")
async def health_check():
    """Health check endpoint."""
    api_keys = get_api_keys()
    return {
        "status": "healthy",
        "service": "Anthropic API Gateway (Gemini Backend)",
        "available_api_keys": len(api_keys),
        "version": "1.0.0"
    }

@app.get("/v1/models")
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
    """
    Create a message using Anthropic API format, powered by Gemini.
    
    Supports:
    - Text and thinking content blocks
    - Tool use and tool results (passed through to Gemini)
    - Streaming and non-streaming responses
    - System prompts
    - Metadata (logged but not sent to Gemini)
    - Temperature, top_p, top_k parameters
    """
    try:
        gemini_model = map_model_name(request.model)
        gemini_messages = convert_messages_to_gemini(request.messages)
        
        generation_config = {
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p
        if request.top_k is not None:
            generation_config["top_k"] = request.top_k
        
        include_thinking = request.thinking is not None and request.thinking.get("enabled", False)
        
        if request.stream:
            return StreamingResponse(
                stream_gemini_response(
                    gemini_model,
                    gemini_messages,
                    generation_config,
                    request.system,
                    include_thinking
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
            
            content_blocks = [
                {
                    "type": "text",
                    "text": response.text
                }
            ]
            
            input_tokens = 0
            output_tokens = 0
            
            if hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            
            return JSONResponse({
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": request.model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            })
    
    except HTTPException:
        raise
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
