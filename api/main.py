
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

# Global state for API key management
api_keys_state = {
    "keys": [],
    "current_index": 0,
    "rate_limited": {},  # key -> timestamp when rate limit expires
    "failed_counts": {}  # key -> consecutive failure count
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
    
    return keys

def initialize_api_keys():
    """Initialize the API keys state."""
    keys = get_api_keys()
    if not keys:
        raise AnthropicError(
            "No GEMINI_API_KEY environment variables configured. Please set GEMINI_API_KEY.",
            error_type="authentication_error",
            status_code=401
        )
    
    api_keys_state["keys"] = keys
    api_keys_state["current_index"] = 0
    api_keys_state["rate_limited"] = {}
    api_keys_state["failed_counts"] = {key: 0 for key in keys}
    
    return keys

class AnthropicError(Exception):
    """Custom exception for Anthropic-compatible errors."""
    def __init__(self, message: str, error_type: str = "api_error", status_code: int = 500):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        super().__init__(self.message)

def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "rate limit", "quota", "429", "resource exhausted", 
        "too many requests", "rate_limit_exceeded"
    ])

def mark_key_rate_limited(api_key: str, duration: int = 60):
    """Mark an API key as rate limited for a specific duration."""
    api_keys_state["rate_limited"][api_key] = time.time() + duration
    api_keys_state["failed_counts"][api_key] = api_keys_state["failed_counts"].get(api_key, 0) + 1
    print(f"[API Key Manager] Key ending in ...{api_key[-6:]} marked as rate limited for {duration}s")

def is_key_available(api_key: str) -> bool:
    """Check if an API key is available (not rate limited)."""
    if api_key in api_keys_state["rate_limited"]:
        if time.time() < api_keys_state["rate_limited"][api_key]:
            return False
        else:
            del api_keys_state["rate_limited"][api_key]
            api_keys_state["failed_counts"][api_key] = 0
            print(f"[API Key Manager] Key ending in ...{api_key[-6:]} is now available again")
    return True

def get_next_available_key() -> str:
    """Get the next available API key that is not rate limited."""
    if not api_keys_state["keys"]:
        initialize_api_keys()
    
    keys = api_keys_state["keys"]
    attempts = 0
    max_attempts = len(keys) * 2
    
    while attempts < max_attempts:
        current_key = keys[api_keys_state["current_index"]]
        api_keys_state["current_index"] = (api_keys_state["current_index"] + 1) % len(keys)
        
        if is_key_available(current_key):
            print(f"[API Key Manager] Using key ending in ...{current_key[-6:]}")
            return current_key
        
        attempts += 1
    
    raise AnthropicError(
        "All API keys are currently rate limited. Please try again later.",
        error_type="rate_limit_error",
        status_code=429
    )

def execute_with_key_rotation(func, *args, **kwargs):
    """Execute a function with automatic key rotation on rate limit errors."""
    keys = api_keys_state["keys"] if api_keys_state["keys"] else initialize_api_keys()
    max_retries = len(keys)
    
    for attempt in range(max_retries):
        try:
            api_key = get_next_available_key()
            genai.configure(api_key=api_key)
            result = func(*args, **kwargs)
            api_keys_state["failed_counts"][api_key] = 0
            return result
        except Exception as e:
            if is_rate_limit_error(e):
                mark_key_rate_limited(api_key, duration=60)
                if attempt < max_retries - 1:
                    print(f"[API Key Manager] Rate limit detected, switching to next key (attempt {attempt + 1}/{max_retries})")
                    continue
                else:
                    raise AnthropicError(
                        "All API keys are rate limited. Please try again later.",
                        error_type="rate_limit_error",
                        status_code=429
                    )
            else:
                raise

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
    """Stream responses from Gemini in Anthropic format with key rotation."""
    def generate_stream():
        model = genai.GenerativeModel(
            model_name=gemini_model_name,
            system_instruction=system_instruction
        )
        return model.generate_content(
            messages,
            generation_config=generation_config,
            stream=True
        )
    
    response = execute_with_key_rotation(generate_stream)
    
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
async def index_page(request: Request):
    """Documentation and information page."""
    keys = api_keys_state["keys"] if api_keys_state["keys"] else initialize_api_keys()
    available_keys = sum(1 for key in keys if is_key_available(key))
    
    # Get the base URL from the request
    base_url = str(request.base_url).rstrip('/')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anthropic API Gateway - Gemini Backend</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}


@app.get("/health")
async def health_check():
    """JSON health check endpoint."""
    keys = api_keys_state["keys"] if api_keys_state["keys"] else initialize_api_keys()
    available_keys = sum(1 for key in keys if is_key_available(key))
    
    return {
        "status": "healthy",
        "service": "Anthropic API Gateway (Gemini Backend)",
        "total_api_keys": len(keys),
        "available_api_keys": available_keys,
        "rate_limited_keys": len(api_keys_state["rate_limited"]),
        "version": "1.1.0"
    }

        .status {{
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        .status-item {{
            text-align: center;
        }}
        .status-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .status-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .section h3 {{
            color: #555;
            margin: 20px 0 10px;
            font-size: 1.3em;
        }}
        .code-block {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            position: relative;
            margin: 15px 0;
        }}
        .code-block pre {{
            margin: 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .copy-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background 0.3s;
        }}
        .copy-btn:hover {{
            background: #5568d3;
        }}
        .copy-btn:active {{
            background: #4a5bbd;
        }}
        .endpoint {{
            background: #e7f3ff;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .endpoint code {{
            background: #d1e7ff;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }}
        .feature-list {{
            list-style: none;
            padding-left: 0;
        }}
        .feature-list li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }}
        .feature-list li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            color: #28a745;
            font-weight: bold;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Anthropic API Gateway</h1>
            <p>Claude-compatible API powered by Google Gemini</p>
        </div>
        
        <div class="status">
            <div class="status-item">
                <div class="status-value">✓</div>
                <div class="status-label">Status: Healthy</div>
            </div>
            <div class="status-item">
                <div class="status-value">{len(keys)}</div>
                <div class="status-label">Total API Keys</div>
            </div>
            <div class="status-item">
                <div class="status-value">{available_keys}</div>
                <div class="status-label">Available Keys</div>
            </div>
            <div class="status-item">
                <div class="status-value">{len(api_keys_state["rate_limited"])}</div>
                <div class="status-label">Rate Limited</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📋 Quick Start</h2>
                <p>This gateway provides an Anthropic-compatible API interface backed by Google Gemini models. No API key required in requests!</p>
                
                <h3>Base URL</h3>
                <div class="code-block">
                    <button class="copy-btn" onclick="copyToClipboard(this, '{base_url}')">Copy</button>
                    <pre>{base_url}</pre>
                </div>
            </div>
            
            <div class="section">
                <h2>🔧 Claude Desktop Configuration</h2>
                <p>Copy and paste this into your <code>~/.claude/settings.json</code> file:</p>
                
                <div class="code-block">
                    <button class="copy-btn" onclick="copyToClipboard(this, claudeConfig)">Copy</button>
                    <pre id="claude-config">{{
  "apiProviders": [
    {{
      "name": "Gemini Gateway",
      "baseUrl": "{base_url}",
      "apiType": "anthropic"
    }}
  ],
  "defaultProvider": "Gemini Gateway",
  "model": "claude-3-5-sonnet-20241022",
  "maxTokens": 4096
}}</pre>
                </div>
                
                <div class="warning">
                    <strong>Note:</strong> On macOS/Linux, the file is at <code>~/.claude/settings.json</code><br>
                    On Windows, it's at <code>%APPDATA%\.claude\settings.json</code>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 API Endpoints</h2>
                
                <div class="endpoint">
                    <strong>POST</strong> <code>/v1/messages</code><br>
                    Create a message (Anthropic-compatible)
                </div>
                
                <div class="endpoint">
                    <strong>GET</strong> <code>/v1/models</code><br>
                    List available models
                </div>
                
                <div class="endpoint">
                    <strong>GET</strong> <code>/health</code><br>
                    Health check (JSON response)
                </div>
            </div>
            
            <div class="section">
                <h2>💬 Example Request</h2>
                <div class="code-block">
                    <button class="copy-btn" onclick="copyToClipboard(this, curlExample)">Copy</button>
                    <pre id="curl-example">curl -X POST {base_url}/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {{"role": "user", "content": "Hello! What can you help me with?"}}
    ]
  }}'</pre>
                </div>
            </div>
            
            <div class="section">
                <h2>✨ Features</h2>
                <ul class="feature-list">
                    <li>Anthropic API compatible endpoints</li>
                    <li>Automatic API key rotation</li>
                    <li>Smart rate limit handling</li>
                    <li>Streaming and non-streaming responses</li>
                    <li>System prompts support</li>
                    <li>Temperature, top_p, top_k controls</li>
                    <li>Multi-turn conversations</li>
                    <li>Token usage reporting</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🤖 Available Models</h2>
                <div class="code-block">
                    <pre>claude-3-5-sonnet-20241022  →  gemini-1.5-pro
claude-3-opus-20240229      →  gemini-1.5-pro
claude-3-sonnet-20240229    →  gemini-1.5-flash
claude-3-haiku-20240307     →  gemini-1.5-flash-8b
claude-3-5-haiku-20241022   →  gemini-1.5-flash</pre>
                </div>
            </div>
            
            <div class="section">
                <h2>📚 Additional Resources</h2>
                <p>For complete documentation including Continue.dev, Cursor, and Cline configurations, see the <a href="https://github.com/your-repo/API_USAGE_GUIDE.md" style="color: #667eea;">API Usage Guide</a>.</p>
            </div>
        </div>
        
        <div class="footer">
            Version 1.1.0 | Powered by Google Gemini | Built with FastAPI
        </div>
    </div>
    
    <script>
        const claudeConfig = document.getElementById('claude-config').textContent;
        const curlExample = document.getElementById('curl-example').textContent;
        
        function copyToClipboard(button, text) {{
            navigator.clipboard.writeText(text).then(() => {{
                const originalText = button.textContent;
                button.textContent = 'Copied!';
                button.style.background = '#28a745';
                setTimeout(() => {{
                    button.textContent = originalText;
                    button.style.background = '#667eea';
                }}, 2000);
            }});
        }}
    </script>
</body>
</html>
    """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

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
    - Automatic API key rotation on rate limits
    
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
            def generate_response():
                model = genai.GenerativeModel(
                    model_name=gemini_model,
                    system_instruction=request.system
                )
                return model.generate_content(
                    gemini_messages,
                    generation_config=generation_config
                )
            
            response = execute_with_key_rotation(generate_response)
            
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
