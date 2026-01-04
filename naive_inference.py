"""
Bare bones single-GPU inference pipeline without KV cache.
Implements basic autoregressive generation.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name="mistralai/Mistral-7B-v0.1"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        print("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


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


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50, benchmark=False):
    """
    Autoregressive generation loop.
    Forward pass → sample token → append to sequence → repeat

    Args:
        benchmark: If True, collect and return timing metrics (TTFT, ITL)

    Returns:
        If benchmark=False: generated_text (str)
        If benchmark=True: (generated_text, metrics_dict)
    """
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    print(f"\nGenerating {max_new_tokens} tokens...")
    print(f"Prompt: {prompt}")
    print("-" * 80)

    # Benchmark metrics
    token_times = []
    start_time = time.perf_counter()

    # Generation loop
    for i in range(max_new_tokens):
        iter_start = time.perf_counter()

        # Forward pass through entire sequence (no KV cache)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Sample next token
        next_token = sample_token(logits, temperature=temperature, top_k=top_k)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        iter_end = time.perf_counter()
        token_times.append(iter_end - iter_start)

        # Decode and print token
        token_text = tokenizer.decode(next_token[0])
        print(token_text, end='', flush=True)

        # Stop if EOS token
        if tokenizer.eos_token_id and next_token.item() == tokenizer.eos_token_id:
            break

    total_time = time.perf_counter() - start_time
    print("\n" + "-" * 80)

    # Collect memory stats
    peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
    peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)  # Convert to GB

    # Decode full sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    if benchmark:
        metrics = compute_metrics(token_times, total_time, len(token_times), peak_memory_allocated, peak_memory_reserved)
        return generated_text, metrics

    return generated_text


def compute_metrics(token_times, total_time, num_tokens, peak_memory_allocated, peak_memory_reserved):
    """
    Compute benchmark metrics.

    Metrics:
        - TTFT (Time To First Token): Latency until first token is generated
        - ITL (Inter-Token Latency): Average time between tokens (excluding first)
        - Throughput: Tokens per second
        - Peak Memory: Maximum GPU memory used during generation
    """
    if len(token_times) == 0:
        return {}

    ttft = token_times[0] * 1000  # Convert to ms

    if len(token_times) > 1:
        sorted_times = sorted(token_times[1:])
        itl_mean = sum(token_times[1:]) / len(token_times[1:]) * 1000
        itl_p50 = sorted_times[len(sorted_times) // 2] * 1000
        itl_p99_idx = min(int(len(sorted_times) * 0.99), len(sorted_times) - 1)
        itl_p99 = sorted_times[itl_p99_idx] * 1000
    else:
        itl_mean = itl_p50 = itl_p99 = 0

    throughput = num_tokens / total_time if total_time > 0 else 0

    metrics = {
        "ttft_ms": ttft,
        "itl_mean_ms": itl_mean,
        "itl_p50_ms": itl_p50,
        "itl_p99_ms": itl_p99,
        "throughput_tokens_per_sec": throughput,
        "total_time_s": total_time,
        "num_tokens": num_tokens,
        "peak_memory_allocated_gb": peak_memory_allocated,
        "peak_memory_reserved_gb": peak_memory_reserved,
    }

    return metrics


def print_metrics(metrics):
    """Write benchmark metrics to a timestamped file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{timestamp}.txt"
    lines = [
        "BENCHMARK METRICS",
        "=" * 80,
        f"Time To First Token (TTFT):        {metrics['ttft_ms']:.2f} ms",
        f"Inter-Token Latency (ITL) - Mean:  {metrics['itl_mean_ms']:.2f} ms",
        f"Inter-Token Latency (ITL) - P50:   {metrics['itl_p50_ms']:.2f} ms",
        f"Inter-Token Latency (ITL) - P99:   {metrics['itl_p99_ms']:.2f} ms",
        f"Throughput:                        {metrics['throughput_tokens_per_sec']:.2f} tokens/sec",
        f"Total Time:                        {metrics['total_time_s']:.2f} s",
        f"Tokens Generated:                  {metrics['num_tokens']}",
        f"Peak Memory Allocated:             {metrics['peak_memory_allocated_gb']:.2f} GB",
        f"Peak Memory Reserved:              {metrics['peak_memory_reserved_gb']:.2f} GB",
    ]
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    # Load model
    model, tokenizer = load_model()

    # Example prompt
    prompt = "The capital of France is"

    # Generate with benchmarking
    output, metrics = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=1000,
        temperature=0.8,
        top_k=50,
        benchmark=True
    )

    print(f"\nFull output:\n{output}")

    # Print benchmark results
    print_metrics(metrics)


if __name__ == "__main__":
    main()
