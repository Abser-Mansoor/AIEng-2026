"""Command-line interface for MiniGPT.

Example:
    python generate.py --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minigpt import count_parameters, generate_text, load_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description="Generate text with the from-scratch MiniGPT model.")
    p.add_argument("--prompt", type=str, default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy decoding")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--checkpoint", type=str, default=os.path.join(HERE, "minigpt_best.pt"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, _ = load_model(args.checkpoint, device=args.device)
    out = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k)

    print(f"Prompt: {args.prompt}")
    print("Generated:")
    print(out)
    print("---")
    print(f"model parameters: {count_parameters(model) / 1e6:.2f}M")
    print(f"context length: {model.cfg.context_length}")
    print(f"temperature: {args.temperature}")
    print(f"top_k: {args.top_k}")
    print(f"seed: {args.seed}")


if __name__ == "__main__":
    main()
