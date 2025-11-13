
# Anthropic API Gateway (Gemini Backend)

This project is a lightweight, high-performance API gateway that translates requests from the Anthropic Claude API format to the Google Gemini API format. It allows you to use Claude-compatible clients and tools (like the Claude Desktop app, Continue.dev, Cursor, and more) with Google's Gemini models seamlessly.

The gateway is built with FastAPI, ensuring high performance and easy deployment. It also includes an intelligent API key rotation system to help you manage rate limits and maximize uptime.

## Key Features

- **Anthropic-Compatible API**: Drop-in replacement for the Anthropic API, allowing you to use your favorite Claude-compatible tools with Gemini models.
- **Intelligent API Key Rotation**: Automatically rotates through a pool of Gemini API keys, gracefully handling rate limits and maximizing availability.
- **Streaming & Non-Streaming Support**: Supports both streaming and non-streaming responses, just like the native Anthropic API.
- **System Prompts**: Pass system prompts to guide the model's behavior.
- **Easy Deployment**: Ready to be deployed on services like Vercel, Replit, or any other platform that supports FastAPI.
- **Health Check Endpoint**: A simple `/` endpoint provides a status page with information about the gateway's health and the status of your API keys.

## Getting Started

### Prerequisites

- Python 3.8+
- An active Google AI Studio account and at least one Gemini API key.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure your API keys:**

    Create a `.env` file in the root of the project and add your Gemini API keys. You can add multiple keys for rotation.

    ```
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

## Basic Usage

Once the server is running, you can make requests to the `/v1/messages` endpoint using any HTTP client or a Claude-compatible tool.

Here's a simple `curl` example:

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello! What can you help me with?"}
    ]
  }'
```

For more detailed information on how to configure various clients and tools, please see the [API Usage Guide](API_USAGE_GUIDE.md).

## Project Structure

- `api/main.py`: The main FastAPI application file containing all the API logic.
- `API_USAGE_GUIDE.md`: A comprehensive guide on how to use the API with various clients and tools.
- `requirements.txt`: A list of the Python dependencies for the project.
- `test_api.py`: A script for testing the API's functionality.
- `vercel.json`: Configuration for deploying the application on Vercel.

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.
