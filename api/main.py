import os
import json
import time
import uuid
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
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

class AnthropicError(Exception):
    """Custom exception for Anthropic-compatible errors."""
    def __init__(self, message: str, error_type: str = "api_error", status_code: int = 500):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        super().__init__(self.message)

def get_next_api_key() -> str:
    """Get the next API key in rotation."""
    global api_key_cycle
    
    api_keys = get_api_keys()
    if not api_keys:
        raise AnthropicError(
            "No GEMINI_API_KEY environment variables configured. Please set GEMINI_API_KEY.",
            error_type="authentication_error",
            status_code=401
        )
    
    if api_key_cycle is None:
        api_key_cycle = cycle(api_keys)
    
    return next(api_key_cycle)

def map_model_name(anthropic_model: str) -> str:
    """Map Anthropic model names to Gemini equivalents."""
    return MODEL_MAPPING.get(anthropic_model, "gemini-1.5-pro")

def parse_content_blocks(content: Union[str, List[ContentBlock]]) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse content blocks and extract text content, tool uses, and tool results."""
    if isinstance(content, str):
        return content, [], []
    
    text_parts = []
    tool_uses = []
    tool_results = []
    
    for block in content:
        block_dict = block if isinstance(block, dict) else block.dict()
        
        if block_dict.get("type") == "text" and "text" in block_dict:
            text_parts.append(block_dict["text"])
        elif block_dict.get("type") == "thinking" and "text" in block_dict:
            text_parts.append(f"[Thinking: {block_dict['text']}]")
        elif block_dict.get("type") == "tool_use":
            tool_uses.append(block_dict)
        elif block_dict.get("type") == "tool_result":
            tool_results.append(block_dict)
    
    return "\n".join(text_parts) if text_parts else "", tool_uses, tool_results

def convert_messages_to_gemini(messages: List[Message]) -> tuple[List[Dict[str, str]], bool, bool]:
    """Convert Anthropic message format to Gemini format.
    
    Returns:
        Tuple of (gemini_messages, has_tool_uses, has_tool_results)
    """
    gemini_messages = []
    has_tool_uses = False
    has_tool_results = False
    
    for msg in messages:
        role = "model" if msg.role == "assistant" else "user"
        
        content_text, tool_uses, tool_results = parse_content_blocks(msg.content)
        
        if tool_uses:
            has_tool_uses = True
        if tool_results:
            has_tool_results = True
        
        if content_text:
            gemini_messages.append({
                "role": role,
                "parts": [content_text]
            })
    
    return gemini_messages, has_tool_uses, has_tool_results

async def stream_gemini_response(
    gemini_model_name: str,
    anthropic_model_name: str,
    messages: List[Dict[str, str]],
    generation_config: Dict[str, Any],
    system_instruction: Optional[str] = None,
    include_thinking: bool = False
) -> AsyncGenerator[str, None]:
    """Stream responses from Gemini in Anthropic format."""
    api_key = get_next_api_key()
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name=gemini_model_name,
        system_instruction=system_instruction
    )
    
    message_start = {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": anthropic_model_name,
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
    
    finish_reason = None
    last_chunk = None
    
    for chunk in response:
        last_chunk = chunk
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
    
    stop_reason = "end_turn"
    stop_sequence = None
    
    if last_chunk and hasattr(last_chunk, 'candidates') and last_chunk.candidates:
        finish_reason = last_chunk.candidates[0].finish_reason
        if finish_reason == 1:
            stop_reason = "end_turn"
        elif finish_reason == 2:
            stop_reason = "max_tokens"
        elif finish_reason in (3, 4, 5, 6):
            stop_reason = "end_turn"
    
    final_event = {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence
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
    - Text content blocks
    - Streaming and non-streaming responses
    - System prompts
    - Metadata (accepted but not sent to Gemini)
    - Temperature, top_p, top_k, stop_sequences parameters
    
    Explicitly Rejected (with clear error messages):
    - Tool definitions (tools parameter)
    - Tool use blocks (tool_use content type)
    - Tool result blocks (tool_result content type)
    - Thinking blocks (thinking parameter)
    """
    try:
        if request.tools:
            raise AnthropicError(
                "Tool definitions are not supported. Gemini does not support Anthropic's tool invocation format. "
                "For tool-based workflows, implement tool execution externally and pass tool_use/tool_result "
                "content blocks in the message history.",
                error_type="invalid_request_error",
                status_code=400
            )
        
        gemini_model = map_model_name(request.model)
        gemini_messages, has_tool_uses, has_tool_results = convert_messages_to_gemini(request.messages)
        
        if has_tool_uses:
            raise AnthropicError(
                "Tool execution is not supported by this gateway. Tool_use content blocks "
                "are detected in your messages, but Gemini cannot execute tools in Anthropic's format. "
                "Please execute tools externally and only send the text results.",
                error_type="invalid_request_error",
                status_code=400
            )
        
        if has_tool_results:
            raise AnthropicError(
                "Tool_result content blocks are not supported by this gateway. "
                "Gemini does not support Anthropic's tool format. "
                "Please execute tools externally and send only the text results in regular messages.",
                error_type="invalid_request_error",
                status_code=400
            )
        
        generation_config = {
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p
        if request.top_k is not None:
            generation_config["top_k"] = request.top_k
        if request.stop_sequences:
            generation_config["stop_sequences"] = request.stop_sequences
        
        include_thinking = request.thinking is not None and request.thinking.get("enabled", False)
        if include_thinking:
            raise AnthropicError(
                "Thinking blocks are not directly supported by Gemini backend. "
                "Thinking content will be passed through message history but not generated by the model.",
                error_type="invalid_request_error",
                status_code=400
            )
        
        if request.stream:
            return StreamingResponse(
                stream_gemini_response(
                    gemini_model,
                    request.model,
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
            
            stop_reason = "end_turn"
            stop_sequence = None
            
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == 1:
                    stop_reason = "end_turn"
                elif finish_reason == 2:
                    stop_reason = "max_tokens"
                elif finish_reason in (3, 4, 5, 6):
                    stop_reason = "end_turn"
            
            return JSONResponse({
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": request.model,
                "stop_reason": stop_reason,
                "stop_sequence": stop_sequence,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            })
    
    except AnthropicError:
        raise
    except Exception as e:
        raise AnthropicError(
            f"Internal server error: {str(e)}",
            error_type="api_error",
            status_code=500
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with Anthropic format."""
    errors = exc.errors()
    error_messages = []
    for error in errors:
        loc = " -> ".join(str(x) for x in error["loc"])
        error_messages.append(f"{loc}: {error['msg']}")
    
    return JSONResponse(
        status_code=400,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": f"Validation error: {'; '.join(error_messages)}"
            }
        }
    )

@app.exception_handler(AnthropicError)
async def anthropic_error_handler(request: Request, exc: AnthropicError):
    """Handle AnthropicError exceptions with proper formatting."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {
                "type": exc.error_type,
                "message": exc.message
            }
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTPException with Anthropic format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": exc.detail
            }
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions in Anthropic-compatible format."""
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
