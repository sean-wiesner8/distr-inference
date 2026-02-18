"""Utilities for computing and reporting inference metrics."""

import time


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
    filename = f"metrics_{metrics['num_tokens']}_{timestamp}.txt"
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
