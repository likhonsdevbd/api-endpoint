# Anthropic API Gateway (Gemini Backend)

## Overview
A FastAPI application that provides an Anthropic API-compatible endpoint using Google's Gemini models as the backend. This allows you to use Gemini models with tools and applications designed for the Anthropic (Claude) API.

## Purpose
- Act as a translation layer between Anthropic API format and Gemini API
- Support multiple Gemini API keys with round-robin rotation
- Enable seamless integration with Anthropic-compatible tools

## Recent Changes
- **2025-11-13**: Initial project setup
  - Created FastAPI application with /anthropic endpoint
  - Implemented model name mapping (Claude → Gemini)
  - Added streaming and non-streaming response support
  - Set up multiple API key rotation
  - Created Vercel deployment configuration

## Project Architecture

### Structure
```
/
├── api/
│   └── main.py          # Main FastAPI application
├── requirements.txt      # Python dependencies
├── vercel.json          # Vercel deployment config
└── .gitignore           # Git ignore rules
```

### Key Components

1. **API Endpoints**:
   - `GET /` - Health check endpoint
   - `GET /models` - List available models
   - `POST /v1/messages` - Anthropic-compatible message endpoint (alias)
   - `POST /anthropic` - Main endpoint for chat completions

2. **Model Mapping**:
   - Claude 3 Opus → Gemini 1.5 Pro
   - Claude 3 Sonnet → Gemini 1.5 Flash
   - Claude 3 Haiku → Gemini 1.5 Flash 8B
   - Claude 3.5 Sonnet → Gemini 1.5 Pro

3. **API Key Rotation**:
   - Supports multiple API keys (GEMINI_API_KEY, GEMINI_API_KEY2, GEMINI_API_KEY3, etc.)
   - Round-robin rotation for load distribution

4. **Features**:
   - Streaming and non-streaming responses
   - System prompts support
   - Temperature, top_p, top_k parameter mapping
   - Anthropic-compatible error responses
   - Token usage reporting

## Environment Variables

Required:
- `GEMINI_API_KEY` - Primary Google AI Studio API key

Optional (for multiple keys):
- `GEMINI_API_KEY2` - Secondary API key
- `GEMINI_API_KEY3` - Tertiary API key
- Additional keys can be added with incrementing numbers

## Deployment

### Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Set environment variables in Vercel dashboard:
   - Add GEMINI_API_KEY with your Google AI Studio key
   - Optionally add GEMINI_API_KEY2, GEMINI_API_KEY3, etc.
3. Deploy: `vercel --prod`

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export GEMINI_API_KEY="your-api-key"

# Run server
uvicorn api.main:app --reload --host 0.0.0.0 --port 5000
```

## Usage Example

```python
import requests

response = requests.post(
    "http://localhost:5000/anthropic",
    json={
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 1024,
        "stream": False
    }
)

print(response.json())
```

## Dependencies
- FastAPI: Web framework
- Uvicorn: ASGI server
- google-generativeai: Gemini SDK
- Pydantic: Data validation
- python-dotenv: Environment management
