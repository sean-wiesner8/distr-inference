"""
Bare bones single-GPU inference pipeline without KV cache.
Implements basic autoregressive generation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name="mistralai/Mistral-7B-v0.1"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    print("Model loaded successfully")
    return model, tokenizer


def sample_token(logits, temperature=1.0, top_k=50):
    """Sample next token from logits."""
    # Get logits for last token
    logits = logits[:, -1, :] / temperature

    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Sample from distribution
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
    """
    Autoregressive generation loop.
    Forward pass → sample token → append to sequence → repeat
    """
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    print(f"\nGenerating {max_new_tokens} tokens...")
    print(f"Prompt: {prompt}")
    print("-" * 80)

    # Generation loop
    for i in range(max_new_tokens):
        # Forward pass through entire sequence (no KV cache)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Sample next token
        next_token = sample_token(logits, temperature=temperature, top_k=top_k)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Decode and print token
        token_text = tokenizer.decode(next_token[0])
        print(token_text, end='', flush=True)

        # Stop if EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

    print("\n" + "-" * 80)

    # Decode full sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    # Load model (use Mistral 7B or change to "meta-llama/Llama-3.2-8B")
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1")

    # Example prompt
    prompt = "The capital of France is"

    # Generate
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    )

    print(f"\nFull output:\n{output}")


if __name__ == "__main__":
    main()
