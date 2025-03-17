from openai import OpenAI
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument("--model_path", type=str, default="", help="Path to the model.")
    parser.add_argument("--prompt", type=str, default="", help="The input prompt.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens of model generation")
    parser.add_argument("--temperature", type=float, default=0.01, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.1, help="top_p")
    parser.add_argument("--timeout", type=int, default=2000, help="timeout")
    args = parser.parse_args()

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": args.prompt,
        },
    ]

    chat_response = client.chat.completions.create(
        model=args.model_path,
        messages=messages,
        stream=False,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
    )

    chat_response = chat_response.choices[0].message.content
    print("Chat response:", chat_response)

if __name__ == "__main__":
    main()