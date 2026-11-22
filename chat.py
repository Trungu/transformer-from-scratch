import torch
import time
from pathlib import Path
from tokenizers import Tokenizer

from model.autoregressive_transformer import AutoRegressiveTransformer

checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

checkpoint = torch.load(
    checkpoint_dir / "tinystories_latest.pt",
    map_location="cpu",
    weights_only=True
)

config = checkpoint["config"]

tokenizer = Tokenizer.from_file(
    str(checkpoint_dir / "tokenizer.json")
)

model = AutoRegressiveTransformer(**config)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

def generate_text(prompt: str, max_new_tokens=150, temperature=0.7, top_p=0.9):

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")

    device = next(model.parameters()).device

    tokens = torch.tensor(
        tokenizer.encode(prompt).ids,
        dtype=torch.long,
        device=device
    )

    max_context_length = model.pos_embeddings.size(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):

            context = tokens[-max_context_length:]
            logits = model(context)

            next_logits = logits[-1] / temperature

            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                # Keep the token that reaches the probability threshold.
                remove_mask = cumulative_probs >= top_p
                remove_mask[1:] = remove_mask[:-1].clone()
                remove_mask[0] = False
                next_logits[sorted_indices[remove_mask]] = float("-inf")

            probabilities = torch.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            tokens = torch.cat((tokens, next_token))

        generated_text = tokenizer.decode(tokens.tolist())

        print("Model: ", end="", flush=True)

        text = generated_text[len(prompt):]

        for char in text:
            print(char, end="", flush=True)
            time.sleep(0.03)

        print()

        return text


def main():
    print("TinyStories Transformer")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")

        if prompt.strip().lower() == "exit":
            break

        if not prompt.strip():
            continue

        generate_text(prompt)
        print()


if __name__ == "__main__":
    main()
