# Anthropic API Gateway (Gemini Backend)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/likhonsdevbd/api-endpoint)

🌐 **Live Demo**: <https://api-only-for-coding.vercel.app>
📦 **GitHub**: <https://github.com/likhonsdevbd/api-endpoint>

This project is a lightweight, high-performance API gateway that translates requests from the Anthropic Claude API format to the Google Gemini API format. It allows you to use Claude-compatible clients and tools (like the Claude Desktop app, Continue.dev, Cursor, Claude Code, and more) with Google’s Gemini models seamlessly.

The gateway is built with FastAPI, ensuring high performance and easy deployment. It also includes an intelligent API key rotation system to help you manage rate limits and maximize uptime.

## ✨ Key Features

- **Anthropic-Compatible API**: Drop-in replacement for the Anthropic API, allowing you to use your favorite Claude-compatible tools with Gemini models.
- **Intelligent API Key Rotation**: Automatically rotates through a pool of Gemini API keys, gracefully handling rate limits and maximizing availability.
- **Streaming & Non-Streaming Support**: Supports both streaming and non-streaming responses, just like the native Anthropic API.
- **System Prompts**: Pass system prompts to guide the model’s behavior.
- **Easy Deployment**: Ready to be deployed on services like Vercel, Replit, or any other platform that supports FastAPI.
- **Health Check Endpoint**: A simple `/` endpoint provides a status page with information about the gateway’s health and the status of your API keys.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An active Google AI Studio account and at least one Gemini API key ([Get yours here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/likhonsdevbd/api-endpoint.git
   cd api-endpoint
   ```
1. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
1. **Configure your API keys:**

   Create a `.env` file in the root of the project and add your Gemini API keys. You can add multiple keys for rotation.

   ```env
   GEMINI_API_KEY="your_gemini_api_key_1"
   GEMINI_API_KEY2="your_gemini_api_key_2"
   GEMINI_API_KEY3="your_gemini_api_key_3"
   ```

### Running Locally

To run the server locally, use `uvicorn`:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`.

## 📖 Basic Usage

### Quick Test with cURL

```bash
curl -X POST https://api-only-for-coding.vercel.app/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello! What can you help me with?"}
    ]
  }'
```

### Python Example

```python
import requests

url = "https://api-only-for-coding.vercel.app/v1/messages"
headers = {
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01"
}

data = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Write a hello world program in Python"}
    ]
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## 🛠️ Using with CLI Tools

### Claude Code (Anthropic’s Official CLI)

[Claude Code](https://docs.claude.com/en/docs/claude-code) is Anthropic’s official command-line tool for agentic coding.

1. **Install Claude Code:**

   ```bash
   npm install -g @anthropic-ai/claude-code
   ```
1. **Configure to use the gateway:**

   Create or edit `~/.config/claude-code/config.json`:

   ```json
   {
     "apiKey": "not-needed",
     "apiUrl": "https://api-only-for-coding.vercel.app"
   }
   ```
1. **Start using Claude Code:**

   ```bash
   claude-code "Create a React component for a todo list"
   ```

### Aider (AI Pair Programming)

[Aider](https://aider.chat/) is a popular AI pair programming tool in your terminal.

1. **Install Aider:**

   ```bash
   pip install aider-chat
   ```
1. **Use with the gateway:**

   ```bash
   aider --api-base https://api-only-for-coding.vercel.app --api-key not-needed --model claude-3-5-sonnet-20241022
   ```
1. **Or set environment variables:**

   ```bash
   export ANTHROPIC_API_BASE=https://api-only-for-coding.vercel.app
   export ANTHROPIC_API_KEY=not-needed
   aider --model claude-3-5-sonnet-20241022
   ```

### Continue.dev (VS Code Extension)

[Continue](https://continue.dev/) is a VS Code extension for AI-powered coding assistance.

1. **Install the Continue extension** in VS Code
1. **Configure in VS Code settings** (`~/.continue/config.json`):

   ```json
   {
     "models": [
       {
         "title": "Gemini (via Gateway)",
         "provider": "anthropic",
         "model": "claude-3-5-sonnet-20241022",
         "apiKey": "not-needed",
         "apiBase": "https://api-only-for-coding.vercel.app"
       }
     ]
   }
   ```

### Cursor IDE

[Cursor](https://cursor.sh/) is an AI-first code editor.

1. **Open Cursor Settings** (Cmd/Ctrl + ,)
1. **Navigate to** “Models” → “Anthropic API”
1. **Configure:**
- API Key: `not-needed`
- Base URL: `https://api-only-for-coding.vercel.app`
- Model: `claude-3-5-sonnet-20241022`

### LangChain CLI

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    anthropic_api_key="not-needed",
    anthropic_api_url="https://api-only-for-coding.vercel.app"
)

response = llm.invoke("Write a function to calculate fibonacci numbers")
print(response.content)
```

### LiteLLM Proxy

[LiteLLM](https://docs.litellm.ai/) provides a unified interface to 100+ LLMs.

```bash
litellm --model anthropic/claude-3-5-sonnet-20241022 \
        --api_base https://api-only-for-coding.vercel.app \
        --api_key not-needed
```

### Shell-GPT

```bash
export OPENAI_API_BASE=https://api-only-for-coding.vercel.app/v1
export OPENAI_API_KEY=not-needed

sgpt "explain docker in simple terms"
```

### Generic Configuration

For any tool that supports Anthropic API:

- **Base URL**: `https://api-only-for-coding.vercel.app`
- **API Key**: `not-needed` (or any string, as the gateway doesn’t require authentication)
- **Model**: Any Claude model name (e.g., `claude-3-5-sonnet-20241022`)

## 🔄 Model Mapping

The gateway automatically maps Claude model names to Gemini models:

|Claude Model                |Gemini Model          |
|----------------------------|----------------------|
|`claude-sonnet-4-20250514`  |`gemini-2.0-flash-exp`|
|`claude-3-5-sonnet-20241022`|`gemini-1.5-pro-002`  |
|`claude-3-5-sonnet-20240620`|`gemini-1.5-pro-002`  |
|`claude-opus-4-20250514`    |`gemini-1.5-pro-002`  |
|`claude-3-opus-20240229`    |`gemini-1.5-pro`      |
|`claude-3-5-haiku-20241022` |`gemini-1.5-flash-002`|
|`claude-3-haiku-20240307`   |`gemini-1.5-flash`    |

You can also use Gemini model names directly (e.g., `gemini-1.5-pro`, `gemini-2.0-flash-exp`).

## 🚢 Deployment

### Deploy to Vercel (Recommended)

1. **Click the deploy button** at the top of this README, or:
1. **Use Vercel CLI:**

   ```bash
   npm i -g vercel
   vercel
   ```
1. **Set environment variables** in Vercel dashboard:
- Go to your project settings
- Add `GEMINI_API_KEY`, `GEMINI_API_KEY2`, `GEMINI_API_KEY3` with your API keys

### Deploy with Docker

```bash
docker build -t anthropic-gemini-gateway .
docker run -p 8000:8000 \
  -e GEMINI_API_KEY="your_key_1" \
  -e GEMINI_API_KEY2="your_key_2" \
  anthropic-gemini-gateway
```

### Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/anthropic-gemini-gateway)

### Deploy to Replit

1. Import the repository into Replit
1. Add your API keys to Secrets
1. Run the application

## 📁 Project Structure

```
api-endpoint/
├── api/
│   └── main.py              # Main FastAPI application
├── API_USAGE_GUIDE.md       # Comprehensive usage guide
├── requirements.txt         # Python dependencies
├── test_api.py             # API testing script
├── vercel.json             # Vercel deployment config
└── README.md               # This file
```

## 🔍 API Endpoints

- `POST /v1/messages` - Create a message (streaming and non-streaming)
- `GET /` - Health check and status page
- `GET /health` - Simple health check

## 🐛 Troubleshooting

### “No API keys configured”

Make sure you’ve set at least one `GEMINI_API_KEY` environment variable.

### Rate Limit Errors

Add more API keys (`GEMINI_API_KEY2`, `GEMINI_API_KEY3`, etc.) to enable automatic rotation.

### Connection Errors in CLI Tools

Ensure you’re using the full URL including `https://` and that there are no trailing slashes.

### Model Not Found

Double-check that you’re using a supported Claude model name from the mapping table above.

## 📚 Additional Resources

- [API Usage Guide](API_USAGE_GUIDE.md) - Detailed examples and advanced usage
- [Google AI Studio](https://aistudio.google.com/app/apikey) - Get your Gemini API keys
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference) - Original API reference
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Framework documentation

## 🤝 Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to:

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Commit your changes (`git commit -m 'Add amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This is an unofficial gateway and is not affiliated with Anthropic or Google. Use responsibly and in accordance with both Anthropic’s and Google’s terms of service. This tool is intended for development and testing purposes.

## 💡 Use Cases

- **Cost Optimization**: Use Gemini’s generous free tier with Claude-compatible tools
- **Testing**: Test your Claude integrations without using Claude API credits
- **Development**: Develop against Claude API format while using Gemini backend
- **Multi-Model Strategy**: Easily switch between Claude and Gemini without code changes

-----

**Made with ❤️ for developers who want flexibility in their AI tooling**

⭐ Star this repo if you find it useful!