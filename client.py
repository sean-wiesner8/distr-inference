"""
Client for making requests to the distributed inference server.
"""

import requests
import json


def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 50,
    server_url: str = "http://localhost:8000"
) -> dict:
    """
    Send a generate request to the inference server.

    Args:
        prompt: The input prompt for text generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        server_url: Base URL of the server

    Returns:
        Response JSON from the server
    """
    endpoint = f"{server_url}/generate"

    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k
    }

    print(f"Sending request to {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 80)

    try:
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        return result

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {server_url}")
        print("Make sure the server is running (python server.py)")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {response.text}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


def main():
    # Request parameters
    prompt = "The capital of France is"
    max_new_tokens = 100
    temperature = 0.8
    top_k = 50

    # Make request to server
    result = generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    # Print results
    print("\nGeneration Result:")
    print("=" * 80)
    print(f"Generated text:\n{result['generated_text']}")
    print("=" * 80)
    print(f"\nParameters used:")
    print(f"  Prompt: {result['prompt']}")
    print(f"  Max new tokens: {result['max_new_tokens']}")
    print(f"  Temperature: {result['temperature']}")
    print(f"  Top-k: {result['top_k']}")


if __name__ == "__main__":
    main()
