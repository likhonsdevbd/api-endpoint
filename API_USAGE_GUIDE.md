
# API Usage Guide

Complete guide for using this Anthropic API Gateway with Claude-compatible tools and CLIs.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Claude Desktop](#claude-desktop)
- [Continue.dev](#continuedev)
- [Cursor](#cursor)
- [Cline (formerly Claude Dev)](#cline)
- [Other Claude-Compatible Tools](#other-claude-compatible-tools)
- [API Features](#api-features)
- [Rate Limiting & Key Rotation](#rate-limiting--key-rotation)

## Quick Start

### Base URL
```
https://your-repl-name.repl.co
```

### Authentication
No API key required in requests - authentication is handled via environment variables on the server.

### Example Request
```bash
curl -X POST https://your-repl-name.repl.co/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Configuration

### Server Setup

1. **Add API Keys** using Replit Secrets:
   - `GEMINI_API_KEY` - Primary key
   - `GEMINI_API_KEY2` - Secondary key (optional)
   - `GEMINI_API_KEY3` - Tertiary key (optional)
   - Continue adding keys with incrementing numbers as needed

2. **Automatic Key Rotation**:
   - The server automatically rotates through available keys
   - Rate-limited keys are temporarily disabled (60 seconds)
   - Switches to next available key when limits are hit

3. **Health Check**:
```bash
curl https://your-repl-name.repl.co/
```

Response shows:
- Total API keys configured
- Currently available (non-rate-limited) keys
- Number of rate-limited keys

## Claude Desktop

### User-Level Settings
Location: `~/.claude/settings.json` (macOS/Linux) or `%APPDATA%\.claude\settings.json` (Windows)

```json
{
  "apiProviders": [
    {
      "name": "Gemini Gateway",
      "baseUrl": "https://your-repl-name.repl.co",
      "apiType": "anthropic"
    }
  ],
  "defaultProvider": "Gemini Gateway"
}
```

### Project-Level Settings
Location: `<project-root>/.claude/settings.json`

```json
{
  "apiProvider": {
    "name": "Gemini Gateway",
    "baseUrl": "https://your-repl-name.repl.co",
    "apiType": "anthropic"
  },
  "model": "claude-3-5-sonnet-20241022",
  "maxTokens": 4096
}
```

## Continue.dev

### Configuration File
Location: `~/.continue/config.json`

```json
{
  "models": [
    {
      "title": "Claude 3.5 Sonnet (Gemini)",
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "apiBase": "https://your-repl-name.repl.co",
      "contextLength": 200000,
      "completionOptions": {
        "maxTokens": 4096,
        "temperature": 0.7
      }
    }
  ],
  "tabAutocompleteModel": {
    "title": "Claude 3 Haiku (Gemini)",
    "provider": "anthropic",
    "model": "claude-3-haiku-20240307",
    "apiBase": "https://your-repl-name.repl.co"
  }
}
```

### VS Code Extension Settings
1. Install Continue extension
2. Open Settings (`Cmd/Ctrl + ,`)
3. Search for "Continue: Config"
4. Edit `config.json` as shown above

## Cursor

### Settings Configuration
Location: Cursor Settings → Models → Custom Models

```json
{
  "models": {
    "claude-3-5-sonnet": {
      "apiBase": "https://your-repl-name.repl.co/v1",
      "provider": "anthropic"
    }
  }
}
```

### .cursorrules File
Create `.cursorrules` in your project root:

```
API Configuration:
- Base URL: https://your-repl-name.repl.co
- Model: claude-3-5-sonnet-20241022
- Max Tokens: 4096

Available Models:
- claude-3-5-sonnet-20241022 (Best quality)
- claude-3-sonnet-20240229 (Balanced)
- claude-3-haiku-20240307 (Fast)
```

## Cline

### Extension Settings
Location: VS Code Settings → Cline

```json
{
  "cline.apiProvider": "anthropic",
  "cline.apiBaseUrl": "https://your-repl-name.repl.co",
  "cline.model": "claude-3-5-sonnet-20241022",
  "cline.maxTokens": 4096
}
```

### Project Configuration
Create `.cline/config.json`:

```json
{
  "apiProvider": "anthropic",
  "apiBaseUrl": "https://your-repl-name.repl.co",
  "model": "claude-3-5-sonnet-20241022",
  "maxTokens": 4096,
  "temperature": 0.7
}
```

## Other Claude-Compatible Tools

### Generic Configuration

Most Claude-compatible tools support these settings:

```json
{
  "provider": "anthropic",
  "baseUrl": "https://your-repl-name.repl.co",
  "apiEndpoint": "https://your-repl-name.repl.co/v1/messages",
  "model": "claude-3-5-sonnet-20241022"
}
```

### Environment Variables

Some tools use environment variables:

```bash
export ANTHROPIC_API_BASE_URL="https://your-repl-name.repl.co"
export ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"
```

### Common Tool Configurations

**Aider:**
```bash
aider --model claude-3-5-sonnet-20241022 \
      --anthropic-api-base https://your-repl-name.repl.co
```

**LangChain:**
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    anthropic_api_url="https://your-repl-name.repl.co/v1"
)
```

**OpenAI SDK (with Anthropic compatibility):**
```python
from anthropic import Anthropic

client = Anthropic(
    base_url="https://your-repl-name.repl.co"
)
```

## API Features

### Supported Features
- ✅ Text messages
- ✅ System prompts
- ✅ Streaming responses
- ✅ Multi-turn conversations
- ✅ Temperature control
- ✅ Top-p and top-k sampling
- ✅ Stop sequences
- ✅ Token usage reporting
- ✅ Automatic API key rotation
- ✅ Rate limit handling

### Not Supported
- ❌ Tool/function calling (execute externally)
- ❌ Vision/image inputs
- ❌ Document/PDF inputs
- ❌ Thinking blocks
- ❌ Cache control

### Available Models

| Anthropic Model | Gemini Backend | Best For |
|----------------|----------------|----------|
| claude-3-5-sonnet-20241022 | gemini-1.5-pro | Complex tasks, best quality |
| claude-3-opus-20240229 | gemini-1.5-pro | Long context, reasoning |
| claude-3-sonnet-20240229 | gemini-1.5-flash | Balanced speed/quality |
| claude-3-haiku-20240307 | gemini-1.5-flash-8b | Fast responses |
| claude-3-5-haiku-20241022 | gemini-1.5-flash | Quick tasks |

## Rate Limiting & Key Rotation

### How It Works

1. **Automatic Detection**: Server detects rate limit errors from Gemini API
2. **Key Marking**: Rate-limited keys are temporarily disabled (60 seconds)
3. **Auto-Switch**: Request automatically retries with next available key
4. **Recovery**: Keys become available again after cooldown period

### Monitoring

Check API health and key status:
```bash
curl https://your-repl-name.repl.co/
```

Response includes:
```json
{
  "status": "healthy",
  "total_api_keys": 3,
  "available_api_keys": 2,
  "rate_limited_keys": 1,
  "version": "1.1.0"
}
```

### Best Practices

1. **Multiple Keys**: Configure 2-3 API keys for redundancy
2. **Monitor Health**: Check `/` endpoint periodically
3. **Handle Errors**: Implement retry logic in your client
4. **Respect Limits**: Space out requests when possible

### Error Responses

Rate limit error (all keys exhausted):
```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "All API keys are currently rate limited. Please try again later."
  }
}
```

## Testing

### Test Non-Streaming
```bash
curl -X POST https://your-repl-name.repl.co/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

### Test Streaming
```bash
curl -X POST https://your-repl-name.repl.co/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ]
  }'
```

### Test System Prompt
```bash
curl -X POST https://your-repl-name.repl.co/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "system": "You are a helpful coding assistant.",
    "messages": [
      {"role": "user", "content": "Explain what a function is"}
    ]
  }'
```

## Support

- **Documentation**: See [README.md](README.md)
- **Issues**: Check server logs for detailed error messages
- **Health Check**: Use `GET /` endpoint to verify server status
- **Model List**: Use `GET /v1/models` to see available models

## Version History

- **v1.1.0**: Added intelligent API key rotation with rate limit handling
- **v1.0.0**: Initial release with Anthropic API compatibility
