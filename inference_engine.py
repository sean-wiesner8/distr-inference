"""
Bare bones single-GPU inference pipeline with KV cache.
Implements basic autoregressive generation with key-value caching.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import compute_metrics


class InferenceEngine:
    """Inference engine for autoregressive generation with KV cache."""

    def __init__(self, model_name="mistralai/Mistral-7B-v0.1"):
        """Load model and tokenizer."""
        print(f"Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=50, benchmark=True):
        """
        Autoregressive generation loop.
        Forward pass → sample token → append to sequence → repeat

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            benchmark: If True, collect and return timing metrics (TTFT, ITL)

        Returns:
            If benchmark=False: generated_text (str)
            If benchmark=True: (generated_text, metrics_dict)
        """
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        all_token_ids = input_ids[0].tolist()

        print(f"\nGenerating {max_new_tokens} tokens...")
        print(f"Prompt: {prompt}")
        print("-" * 80)

        # Benchmark metrics
        token_times = []
        start_time = time.perf_counter()

        # KV cache
        past_key_values = None

        # Generation loop
        for i in range(max_new_tokens):
            iter_start = time.perf_counter()

            # Forward pass with KV cache
            with torch.no_grad():
                outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits
                past_key_values = outputs.past_key_values

            # Sample next token
            next_token = self.sample_token(logits, temperature=temperature, top_k=top_k)

            # Update for next iteration
            input_ids = next_token
            all_token_ids.append(next_token.item())

            iter_end = time.perf_counter()
            token_times.append(iter_end - iter_start)

            # Decode and print token
            token_text = self.tokenizer.decode(next_token[0])
            print(token_text, end='', flush=True)

            # Stop if EOS token
            if self.tokenizer.eos_token_id and next_token.item() == self.tokenizer.eos_token_id:
                break

        total_time = time.perf_counter() - start_time
        print("\n" + "-" * 80)

        # Collect memory stats
        peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
        peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)  # Convert to GB

        # Decode full sequence
        generated_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)

        if benchmark:
            metrics = compute_metrics(token_times, total_time, len(token_times), peak_memory_allocated, peak_memory_reserved)
            return generated_text, metrics

        return generated_text

    @staticmethod
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

