"""
Example usage of the Anthropic API Gateway (Gemini Backend)

This script demonstrates how to use the API with different features:
1. Basic non-streaming request
2. Streaming request
3. Multi-turn conversation with message history
4. Thinking blocks support
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def example_basic_request():
    """Example 1: Basic non-streaming request"""
    print("=== Example 1: Basic Request ===")
    
    response = requests.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello! What's 2+2?"}
            ]
        }
    )
    
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}\n")
    return result

def example_streaming_request():
    """Example 2: Streaming request"""
    print("=== Example 2: Streaming Request ===")
    
    response = requests.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Count from 1 to 5"}
            ]
        },
        stream=True
    )
    
    print("Streaming response:")
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data = line_str[6:]
                if data and data != '{}':
                    print(f"  {data}")
    print()

def example_multi_turn_conversation():
    """Example 3: Multi-turn conversation maintaining message history"""
    print("=== Example 3: Multi-turn Conversation ===")
    
    messages = [
        {"role": "user", "content": "My name is Alice. Remember this."}
    ]
    
    response1 = requests.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": messages
        }
    )
    
    result1 = response1.json()
    print(f"Turn 1 - User: {messages[0]['content']}")
    print(f"Turn 1 - Assistant: {result1['content'][0]['text']}\n")
    
    messages.append({
        "role": "assistant",
        "content": result1["content"]
    })
    
    messages.append({
        "role": "user",
        "content": "What's my name?"
    })
    
    response2 = requests.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": messages
        }
    )
    
    result2 = response2.json()
    print(f"Turn 2 - User: {messages[2]['content']}")
    print(f"Turn 2 - Assistant: {result2['content'][0]['text']}\n")

def example_with_system_prompt():
    """Example 4: Using system prompts"""
    print("=== Example 4: System Prompt ===")
    
    response = requests.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "system": "You are a helpful assistant that always responds in a cheerful and encouraging tone.",
            "messages": [
                {"role": "user", "content": "I'm learning to code."}
            ]
        }
    )
    
    result = response.json()
    print(f"Response: {result['content'][0]['text']}\n")

def example_with_metadata():
    """Example 5: Using metadata (logged but not sent to Gemini)"""
    print("=== Example 5: With Metadata ===")
    
    response = requests.post(
        f"{BASE_URL}/v1/messages",
        json={
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "metadata": {
                "user_id": "user_12345"
            },
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }
    )
    
    result = response.json()
    print(f"Response: {result['content'][0]['text']}\n")

def list_available_models():
    """List all available models"""
    print("=== Available Models ===")
    
    response = requests.get(f"{BASE_URL}/v1/models")
    result = response.json()
    
    for model in result['data']:
        print(f"  - {model['id']} → {model['backend_model']}")
    print()

if __name__ == "__main__":
    print("Anthropic API Gateway (Gemini Backend) - Usage Examples\n")
    
    list_available_models()
    
    example_basic_request()
    
    example_streaming_request()
    
    example_multi_turn_conversation()
    
    example_with_system_prompt()
    
    example_with_metadata()
    
    print("=== All examples completed! ===")
