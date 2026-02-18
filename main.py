from inference_engine import InferenceEngine
from metrics import print_metrics


def main():
    """Main entry point for running inference."""
    engine = InferenceEngine()

    prompt = "The capital of France is"
    max_new_tokens = 1000
    temperature = 0.8
    top_k = 50

    generated_text, metrics = engine.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        benchmark=True
    )

    print(generated_text)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
