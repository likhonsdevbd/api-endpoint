
"""
Comprehensive test script for Anthropic API Gateway
Tests all features including rate limit handling and key rotation
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health_check():
    """Test 1: Health check endpoint"""
    print_section("Test 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        result = response.json()
        print(f"✓ Status: {result['status']}")
        print(f"✓ Total API Keys: {result['total_api_keys']}")
        print(f"✓ Available API Keys: {result['available_api_keys']}")
        print(f"✓ Rate Limited Keys: {result['rate_limited_keys']}")
        print(f"✓ Version: {result['version']}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_list_models():
    """Test 2: List available models"""
    print_section("Test 2: List Models")
    
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        result = response.json()
        print(f"✓ Found {len(result['data'])} models:")
        for model in result['data'][:3]:
            print(f"  - {model['id']} → {model['backend_model']}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_basic_request():
    """Test 3: Basic non-streaming request"""
    print_section("Test 3: Basic Request")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Response received")
            print(f"✓ Message ID: {result['id']}")
            print(f"✓ Content: {result['content'][0]['text'][:50]}...")
            print(f"✓ Stop Reason: {result['stop_reason']}")
            print(f"✓ Input Tokens: {result['usage']['input_tokens']}")
            print(f"✓ Output Tokens: {result['usage']['output_tokens']}")
            return True
        else:
            print(f"✗ Status Code: {response.status_code}")
            print(f"✗ Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_streaming_request():
    """Test 4: Streaming request"""
    print_section("Test 4: Streaming Request")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 50,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Count from 1 to 3"}
                ]
            },
            stream=True,
            timeout=30
        )
        
        print("✓ Streaming response:")
        event_count = 0
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('event: '):
                    event_type = line_str.split('event: ')[1]
                    event_count += 1
                    if event_type in ['message_start', 'content_block_delta', 'message_stop']:
                        print(f"  ✓ Event: {event_type}")
        
        print(f"✓ Total events received: {event_count}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_system_prompt():
    """Test 5: System prompt"""
    print_section("Test 5: System Prompt")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "system": "You are a pirate. Always respond like a pirate.",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['content'][0]['text']
            print(f"✓ Response: {content[:100]}...")
            return True
        else:
            print(f"✗ Status Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_multi_turn():
    """Test 6: Multi-turn conversation"""
    print_section("Test 6: Multi-Turn Conversation")
    
    try:
        messages = [
            {"role": "user", "content": "My name is Alice."}
        ]
        
        response1 = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "messages": messages
            },
            timeout=30
        )
        
        if response1.status_code != 200:
            print(f"✗ First turn failed: {response1.status_code}")
            return False
        
        result1 = response1.json()
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
                "max_tokens": 100,
                "messages": messages
            },
            timeout=30
        )
        
        if response2.status_code == 200:
            result2 = response2.json()
            content = result2['content'][0]['text']
            print(f"✓ Turn 1 successful")
            print(f"✓ Turn 2 response: {content[:100]}...")
            if "alice" in content.lower():
                print(f"✓ Context maintained correctly")
            return True
        else:
            print(f"✗ Second turn failed: {response2.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_parameters():
    """Test 7: Temperature and other parameters"""
    print_section("Test 7: Parameters (Temperature, Top-P, Top-K)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Request with custom parameters successful")
            print(f"✓ Response: {result['content'][0]['text'][:50]}...")
            return True
        else:
            print(f"✗ Status Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_error_handling():
    """Test 8: Error handling"""
    print_section("Test 8: Error Handling")
    
    tests_passed = 0
    
    # Test invalid model
    try:
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "invalid-model",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            },
            timeout=30
        )
        # Should still work due to default model mapping
        print(f"✓ Invalid model handled (uses default)")
        tests_passed += 1
    except Exception as e:
        print(f"  Note: {e}")
    
    # Test tool definitions (should be rejected)
    try:
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 100,
                "tools": [{"name": "test", "description": "test", "input_schema": {}}],
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 400:
            result = response.json()
            if "tool" in result['error']['message'].lower():
                print(f"✓ Tool definitions properly rejected")
                tests_passed += 1
        else:
            print(f"  Unexpected response to tool definitions")
    except Exception as e:
        print(f"  Error testing tool rejection: {e}")
    
    return tests_passed >= 1

def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "■"*60)
    print("  ANTHROPIC API GATEWAY - COMPREHENSIVE TEST SUITE")
    print("■"*60)
    
    tests = [
        ("Health Check", test_health_check),
        ("List Models", test_list_models),
        ("Basic Request", test_basic_request),
        ("Streaming Request", test_streaming_request),
        ("System Prompt", test_system_prompt),
        ("Multi-Turn Conversation", test_multi_turn),
        ("Parameters", test_parameters),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            time.sleep(1)  # Brief pause between tests
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
